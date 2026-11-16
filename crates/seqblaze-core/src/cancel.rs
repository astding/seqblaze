//! Per-call cancel scope: Ctrl-C cancellation (D86 (3), (6)) and `progress=True`
//! (D48). Spec: docs/wiki/engine/cancellation-progress.md.
//!
//! A binding creates one [`Scope`] per public call and enters it as the first
//! statement of every GIL-released region of that call, with the caller's
//! floating-point state word that the region's outermost guard saved. Spawn
//! sites capture the calling thread's scope ([`current`]) and enter it on the
//! spawned thread ([`unit`]). Engine loops report finished items through a
//! per-unit [`Ticker`], which makes one unit checkpoint per [`CHUNK`] items and
//! one for the remainder; the calling thread also reports [`boundary`]
//! checkpoints between host steps. A checkpoint reads the scope's flag and
//! returns [`EngineError::Cancelled`] once it is set.
//!
//! Only the calling thread polls (§3.3): at its own checkpoints, at boundaries
//! and while it waits for spawned units ([`recv`]), at most every
//! [`POLL_INTERVAL`]. A poll writes back the caller's state word, lets the
//! binding's [`Watcher`] check signals and write a progress line, then
//! installs E0 again. Outside every scope a checkpoint does nothing but read
//! one thread-local.
//!
//! This module does not depend on Python: the binding supplies the watcher.
use std::{
    cell::RefCell,
    fmt::Write as _,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        mpsc::{Receiver, RecvError, RecvTimeoutError},
        Arc, Mutex,
    },
    thread::ThreadId,
    time::{Duration, Instant},
};

use crate::{fpenv, validate::EngineError};

/// Items per unit checkpoint (D86 (3): at most 4096).
pub const CHUNK: usize = 4096;
/// Minimum time between two polls of the calling thread.
pub const POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Minimum time from the call's start to its first progress line, and between lines.
pub const PROGRESS_INTERVAL: Duration = Duration::from_millis(1000);

/// Checkpoint site keys (§3.2, §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Site {
    Validate = 0,
    Bound = 1,
    Pairs = 2,
    TopK1 = 3,
    TopK2 = 4,
    Details = 5,
    Sequences = 6,
    Tile = 7,
    Routed = 8,
    Continuation = 9,
}

impl Site {
    /// Every site, in key order.
    pub const ALL: [Site; 10] = [
        Site::Validate,
        Site::Bound,
        Site::Pairs,
        Site::TopK1,
        Site::TopK2,
        Site::Details,
        Site::Sequences,
        Site::Tile,
        Site::Routed,
        Site::Continuation,
    ];

    /// The site key of the testing probe.
    pub fn key(self) -> &'static str {
        match self {
            Site::Validate => "validate",
            Site::Bound => "bound",
            Site::Pairs => "pairs",
            Site::TopK1 => "topk1",
            Site::TopK2 => "topk2",
            Site::Details => "details",
            Site::Sequences => "sequences",
            Site::Tile => "tile",
            Site::Routed => "routed",
            Site::Continuation => "continuation",
        }
    }

    /// The site of a key, if any.
    pub fn from_key(key: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.key() == key)
    }
}

/// Progress stages (§4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Validate,
    Bound,
    Pairs,
    TopK1,
    TopK2,
    Details,
    Sequences,
    /// Tile `index` (1-based) of `of` tiles in the grid.
    Tile {
        index: usize,
        of: usize,
    },
}

impl Stage {
    /// The checkpoint key whose items count toward this stage.
    pub fn site(self) -> Site {
        match self {
            Stage::Validate => Site::Validate,
            Stage::Bound => Site::Bound,
            Stage::Pairs => Site::Pairs,
            Stage::TopK1 => Site::TopK1,
            Stage::TopK2 => Site::TopK2,
            Stage::Details => Site::Details,
            Stage::Sequences => Site::Sequences,
            Stage::Tile { .. } => Site::Tile,
        }
    }

    /// The label of the progress line.
    pub fn label(self) -> String {
        match self {
            Stage::Validate => "validate".into(),
            Stage::Bound => "bound".into(),
            Stage::Pairs => "pairs".into(),
            Stage::TopK1 => "topk pass 1/2".into(),
            Stage::TopK2 => "topk pass 2/2".into(),
            Stage::Details => "details".into(),
            Stage::Sequences => "sequences".into(),
            Stage::Tile { index, of } => format!("tile {index}/{of}"),
        }
    }

    /// The unit of the stage's counts.
    pub fn unit(self) -> &'static str {
        match self {
            Stage::Validate | Stage::Sequences => "sequences",
            Stage::Bound => "rows",
            Stage::Pairs | Stage::TopK1 | Stage::TopK2 | Stage::Tile { .. } => "pairs",
            Stage::Details => "entries",
        }
    }
}

/// One opened stage of a call: label, unit, total and done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRow {
    pub label: String,
    pub unit: &'static str,
    pub total: u64,
    pub done: u64,
}

/// What a poll did (returned by the binding's [`Watcher`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PollOutcome {
    /// The poll attached to the interpreter.
    pub attached: bool,
    /// A handler or a progress write raised an exception the call must raise: cancel.
    pub cancel: bool,
    /// The progress line was written.
    pub line_written: bool,
    /// Progress is switched off for the rest of the call (a failed write).
    pub progress_off: bool,
}

/// The binding's side of a poll (§3.3 steps 2-4). It runs on the calling thread,
/// in the caller's floating-point state, and must not wait on anything a unit
/// could hold.
pub trait Watcher: Send + Sync {
    /// Check signals and, when `line` is given, write it.
    fn poll(&self, line: Option<&str>) -> PollOutcome;
}

const NO_STAGE: u8 = u8::MAX;

/// Calling-thread state of a call, behind a lock that is never held across a poll.
struct CallState {
    last_poll: Instant,
    last_line: Instant,
    stages: Vec<StageRow>,
}

/// The cancel scope of one call (§3.1).
pub struct Scope {
    flag: AtomicBool,
    watcher: Arc<dyn Watcher>,
    /// The public kernel name of progress lines; None without progress.
    name: Option<Box<str>>,
    progress: AtomicBool,
    started: Instant,
    calling: ThreadId,
    stage_site: AtomicU8,
    done: AtomicU64,
    state: Mutex<CallState>,
    ended: AtomicBool,
}

impl Scope {
    /// A new scope for a call made on this thread. `progress` is the public
    /// kernel name when progress lines are on.
    pub fn new(watcher: Arc<dyn Watcher>, progress: Option<&str>) -> Arc<Scope> {
        let now = Instant::now();
        Arc::new(Scope {
            flag: AtomicBool::new(false),
            watcher,
            name: progress.map(Into::into),
            progress: AtomicBool::new(progress.is_some()),
            started: now,
            calling: std::thread::current().id(),
            stage_site: AtomicU8::new(NO_STAGE),
            done: AtomicU64::new(0),
            state: Mutex::new(CallState {
                last_poll: now,
                last_line: now,
                stages: Vec::new(),
            }),
            ended: AtomicBool::new(false),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CallState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the cancel flag is set.
    pub fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Enter this scope on the calling thread for one GIL-released region.
    /// `word` is the caller's state that the region's outermost guard saved.
    pub fn enter(self: &Arc<Self>, word: fpenv::State) -> Entered {
        let calling = std::thread::current().id() == self.calling;
        let depth = push(Entry {
            scope: Arc::clone(self),
            word: calling.then_some(word),
        });
        Entered {
            depth,
            _thread: std::marker::PhantomData,
        }
    }

    /// End the call: record its stages and its outcome (once).
    pub fn end(&self) {
        if self.ended.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut state = self.lock();
        if let Some(last) = state.stages.last_mut() {
            last.done = self.done.load(Ordering::SeqCst);
        }
        self.stage_site.store(NO_STAGE, Ordering::SeqCst);
        #[cfg(feature = "testing")]
        probe::ended(self.cancelled(), &state.stages);
        #[cfg(not(feature = "testing"))]
        let _ = &state;
    }

    /// The stages opened so far, with the open stage's current count.
    pub fn stages(&self) -> Vec<StageRow> {
        let mut rows = self.lock().stages.clone();
        if !self.ended.load(Ordering::SeqCst) {
            if let Some(last) = rows.last_mut() {
                last.done = self.done.load(Ordering::SeqCst);
            }
        }
        rows
    }

    fn open(&self, stage: Stage, total: usize) {
        let mut state = self.lock();
        let previous = self.done.swap(0, Ordering::SeqCst);
        if let Some(last) = state.stages.last_mut() {
            last.done = previous;
        }
        state.stages.push(StageRow {
            label: stage.label(),
            unit: stage.unit(),
            total: total as u64,
            done: 0,
        });
        self.stage_site.store(stage.site() as u8, Ordering::SeqCst);
    }

    fn count(&self, site: Site, items: usize) {
        if self.stage_site.load(Ordering::Relaxed) == site as u8 {
            self.done.fetch_add(items as u64, Ordering::Relaxed);
        }
    }

    /// Time left until the next poll is due (at least 1 ms); None after the flag is set.
    fn until_poll(&self) -> Option<Duration> {
        if self.cancelled() {
            return None;
        }
        let elapsed = self.lock().last_poll.elapsed();
        Some(
            POLL_INTERVAL
                .saturating_sub(elapsed)
                .max(Duration::from_millis(1)),
        )
    }

    /// The progress line of the open stage, if one is open.
    fn line(&self, state: &CallState, now: Instant) -> Option<String> {
        let name = self.name.as_deref()?;
        let row = state.stages.last()?;
        let done = self.done.load(Ordering::Relaxed);
        let pct = (u128::from(done) * 100 / u128::from(row.total.max(1))) as u64;
        let secs = now.duration_since(self.started).as_secs();
        let mut line = String::new();
        let _ = writeln!(
            line,
            "seqblaze {name}: {} {done}/{} {} ({pct}%), {secs} s",
            row.label, row.total, row.unit
        );
        Some(line)
    }

    /// A poll of the calling thread if one is due (§3.3).
    fn poll_if_due(&self, word: fpenv::State) {
        if self.cancelled() {
            return;
        }
        let now = Instant::now();
        let line = {
            let mut state = self.lock();
            if now.duration_since(state.last_poll) < POLL_INTERVAL {
                return;
            }
            state.last_poll = now;
            if self.progress.load(Ordering::Relaxed)
                && now.duration_since(state.last_line) >= progress_interval()
            {
                self.line(&state, now)
            } else {
                None
            }
        };
        // Python code runs in the caller's environment; engine work resumes in E0.
        // SAFETY: `word` was read from this thread's register by the region's guard.
        unsafe { fpenv::write_state(word) };
        // Python code run by the poll sees no scope (§3.3): a native step it calls
        // without a call of its own behaves as outside every scope, and a nested
        // call pushes its own scope onto an empty stack. The stack comes back
        // even if the poll unwinds.
        let outcome = {
            let _masked = Masked::new();
            self.watcher.poll(line.as_deref())
        };
        // SAFETY: E0 is a valid state on every target.
        unsafe { fpenv::write_state(fpenv::E0) };
        if outcome.line_written {
            self.lock().last_line = now;
        }
        #[cfg(feature = "testing")]
        probe::polled(outcome.attached, outcome.line_written);
        if outcome.progress_off {
            self.progress.store(false, Ordering::Relaxed);
        }
        if outcome.cancel {
            self.flag.store(true, Ordering::SeqCst);
        }
    }
}

/// One entered scope on a thread's stack.
#[derive(Clone)]
struct Entry {
    scope: Arc<Scope>,
    /// The caller's state word: Some on the calling thread (which polls), None in units.
    word: Option<fpenv::State>,
}

thread_local! {
    static STACK: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

fn push(entry: Entry) -> usize {
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push(entry);
        stack.len()
    })
}

fn pop(depth: usize) {
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        debug_assert_eq!(stack.len(), depth, "cancel scopes leave in reverse order");
        stack.truncate(depth.saturating_sub(1));
    })
}

/// This thread's scope stack, set aside while a poll runs Python code and put
/// back when dropped.
struct Masked {
    saved: Vec<Entry>,
}

impl Masked {
    fn new() -> Self {
        Self {
            saved: STACK.with(|stack| std::mem::take(&mut *stack.borrow_mut())),
        }
    }
}

impl Drop for Masked {
    fn drop(&mut self) {
        let saved = std::mem::take(&mut self.saved);
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            debug_assert!(stack.is_empty(), "a nested call left its scope entered");
            *stack = saved;
        });
    }
}

/// The innermost entered scope of this thread (cloned: no borrow is held).
fn top() -> Option<Entry> {
    STACK.with(|stack| stack.borrow().last().cloned())
}

/// A region's entry into its call's scope; leaves it when dropped.
#[must_use = "the scope is left when this value is dropped"]
pub struct Entered {
    depth: usize,
    _thread: std::marker::PhantomData<*const ()>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        pop(self.depth);
    }
}

/// The calling thread's current scope, for a spawn site to hand to its units.
pub fn current() -> Option<Arc<Scope>> {
    top().map(|entry| entry.scope)
}

/// Whether this thread is inside a scope whose flag is set.
pub fn is_cancelled() -> bool {
    top().is_some_and(|entry| entry.scope.cancelled())
}

/// `Err(Cancelled)` when this thread's scope has its flag set.
pub fn check() -> Result<(), EngineError> {
    if is_cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

/// A running scheduler unit (spawned or inline); leaves its scope when dropped.
#[must_use = "the unit ends when this value is dropped"]
pub struct Unit {
    depth: Option<usize>,
    live: bool,
    _thread: std::marker::PhantomData<*const ()>,
}

impl Drop for Unit {
    fn drop(&mut self) {
        if let Some(depth) = self.depth {
            pop(depth);
        }
        #[cfg(feature = "testing")]
        if self.live {
            probe::unit_ended();
        }
        #[cfg(not(feature = "testing"))]
        let _ = self.live;
    }
}

/// Enter `scope` (captured by [`current`] on the calling thread) as a spawned unit.
pub fn unit(scope: Option<Arc<Scope>>) -> Unit {
    let live = scope.is_some();
    #[cfg(feature = "testing")]
    if live {
        probe::unit_started();
    }
    let depth = scope.map(|scope| push(Entry { scope, word: None }));
    Unit {
        depth,
        live,
        _thread: std::marker::PhantomData,
    }
}

/// An inline unit on this thread (it uses the thread's own scope).
pub fn inline_unit() -> Unit {
    let live = STACK.with(|stack| !stack.borrow().is_empty());
    #[cfg(feature = "testing")]
    if live {
        probe::unit_started();
    }
    Unit {
        depth: None,
        live,
        _thread: std::marker::PhantomData,
    }
}

/// Open a progress stage on the calling thread (§4.2). A stage of 0 items is not opened.
pub fn stage(stage: Stage, total: usize) {
    if total == 0 {
        return;
    }
    if let Some(entry) = top() {
        if entry.word.is_some() {
            entry.scope.open(stage, total);
        }
    }
}

/// A unit checkpoint: `items` finished since the previous one (§3.2).
#[inline(never)]
pub fn checkpoint(site: Site, items: usize) -> Result<(), EngineError> {
    reach(site, items, true)
}

/// A poll point outside every checkpoint site: steps 3 and 4 of a
/// checkpoint (a due poll on the calling thread, then the flag), with no
/// counter and no testing probe. For passes that precede the validation
/// stage and replace no counted work of their own (the stream pass of
/// `SeqTable::stream_certify`).
pub fn poll() -> Result<(), EngineError> {
    let Some(entry) = top() else {
        return Ok(());
    };
    if let Some(word) = entry.word {
        entry.scope.poll_if_due(word);
    }
    if entry.scope.cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

/// A boundary on the calling thread between host steps (§3.5).
#[inline(never)]
pub fn boundary(site: Site, items: usize) -> Result<(), EngineError> {
    reach(site, items, false)
}

fn reach(site: Site, items: usize, unit: bool) -> Result<(), EngineError> {
    let Some(entry) = top() else {
        return Ok(());
    };
    let scope = &entry.scope;
    // 1. Counters.
    scope.count(site, items);
    // 2. The testing probe.
    #[cfg(feature = "testing")]
    if let Some(timeout_ms) = probe::reached(site, items, unit) {
        probe::pause(scope, entry.word, timeout_ms)?;
    }
    #[cfg(not(feature = "testing"))]
    let _ = unit;
    // 3. A due poll on the calling thread.
    if let Some(word) = entry.word {
        scope.poll_if_due(word);
    }
    // 4. The flag.
    if scope.cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

/// Receive from a unit channel on the coordinating thread. On the calling
/// thread of a scope the wait is timed and polls whenever a poll is due (§3.4);
/// elsewhere it is a plain `recv`.
pub fn recv<T>(receiver: &Receiver<T>) -> Result<T, RecvError> {
    let Some(Entry {
        scope,
        word: Some(word),
    }) = top()
    else {
        return receiver.recv();
    };
    loop {
        let Some(wait) = scope.until_poll() else {
            return receiver.recv();
        };
        match receiver.recv_timeout(wait) {
            Ok(value) => return Ok(value),
            Err(RecvTimeoutError::Disconnected) => return Err(RecvError),
            Err(RecvTimeoutError::Timeout) => scope.poll_if_due(word),
        }
    }
}

/// Wait on the calling thread of a scope while `busy()` holds, polling whenever
/// a poll is due (§3.5: a CUDA host wait on a stream). It counts no checkpoint.
/// The wait is never cut short: once the flag is set, and always outside a
/// scope or on a thread that does not poll, this returns at once and the
/// caller's own blocking wait then runs to completion.
pub fn wait_while(mut busy: impl FnMut() -> bool) {
    let Some(Entry {
        scope,
        word: Some(word),
    }) = top()
    else {
        return;
    };
    let Some(wait) = scope.until_poll() else {
        return;
    };
    let mut due = Instant::now() + wait;
    while busy() {
        if Instant::now() >= due {
            scope.poll_if_due(word);
            let Some(wait) = scope.until_poll() else {
                return;
            };
            due = Instant::now() + wait;
        }
        std::thread::yield_now();
    }
}

/// Sends one completion signal when dropped (normal return, error or unwind).
pub struct Done(pub std::sync::mpsc::Sender<()>);

impl Drop for Done {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// Wait until `n` units have signalled through [`Done`] (or every sender is
/// gone), polling as [`recv`] does.
pub fn wait_done(receiver: &Receiver<()>, n: usize) {
    for _ in 0..n {
        if recv(receiver).is_err() {
            break;
        }
    }
}

/// Per-unit item counter: one checkpoint per [`CHUNK`] items, one for the rest.
pub struct Ticker {
    site: Site,
    n: usize,
}

impl Ticker {
    #[inline]
    pub fn new(site: Site) -> Self {
        Self { site, n: 0 }
    }

    /// One item finished.
    #[inline(always)]
    pub fn tick(&mut self) -> Result<(), EngineError> {
        self.n += 1;
        if self.n >= CHUNK {
            self.flush()
        } else {
            Ok(())
        }
    }

    /// `items` items finished (several at once, at most [`CHUNK`] between checkpoints).
    #[inline]
    pub fn add(&mut self, items: usize) -> Result<(), EngineError> {
        let mut items = items;
        while self.n + items >= CHUNK {
            let take = CHUNK - self.n;
            self.n = CHUNK;
            items -= take;
            self.flush()?;
        }
        self.n += items;
        Ok(())
    }

    #[cold]
    fn flush(&mut self) -> Result<(), EngineError> {
        let n = std::mem::take(&mut self.n);
        checkpoint(self.site, n)
    }

    /// The end of the unit: a checkpoint for any remainder.
    #[inline]
    pub fn finish(&mut self) -> Result<(), EngineError> {
        if self.n > 0 {
            self.flush()
        } else {
            Ok(())
        }
    }
}

/// The progress interval (the testing probe may replace it).
fn progress_interval() -> Duration {
    #[cfg(feature = "testing")]
    if let Some(interval) = probe::interval() {
        return interval;
    }
    PROGRESS_INTERVAL
}

/// Process-wide testing probe (§7): counts checkpoints and pauses one of them.
#[cfg(feature = "testing")]
pub mod probe {
    use super::*;
    use std::sync::atomic::AtomicI64;

    struct Config {
        /// None: disarmed; Some(None): any key; Some(Some(site)): that key.
        stage: Option<Option<Site>>,
        nth: u64,
        timeout_ms: u64,
        interval: Option<Duration>,
    }

    static CONFIG: Mutex<Config> = Mutex::new(Config {
        stage: None,
        nth: 1,
        timeout_ms: 10_000,
        interval: None,
    });
    static MATCHED: AtomicU64 = AtomicU64::new(0);
    static FIRED: AtomicBool = AtomicBool::new(false);
    static PAUSED: AtomicBool = AtomicBool::new(false);
    static PAUSES: AtomicU64 = AtomicU64::new(0);
    static RELEASE: AtomicU64 = AtomicU64::new(0);
    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    static CHECKPOINTS: [AtomicU64; 10] = [ZERO; 10];
    static ITEMS: [AtomicU64; 10] = [ZERO; 10];
    static MAX_GAP: AtomicU64 = AtomicU64::new(0);
    static POLLS: AtomicU64 = AtomicU64::new(0);
    static CANCELLED: AtomicU64 = AtomicU64::new(0);
    static LIVE: AtomicI64 = AtomicI64::new(0);
    static LINES: AtomicU64 = AtomicU64::new(0);
    static STAGES: Mutex<Vec<StageRow>> = Mutex::new(Vec::new());

    fn config() -> std::sync::MutexGuard<'static, Config> {
        CONFIG.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replace the configuration, reset every statistic, release a paused unit.
    /// `stage`: None (disarm), Some(None) (any key) or Some(Some(site)).
    pub fn arm(stage: Option<Option<Site>>, nth: u64, timeout_ms: u64, interval: Option<Duration>) {
        let mut cfg = config();
        cfg.stage = stage;
        cfg.nth = nth.max(1);
        cfg.timeout_ms = timeout_ms.max(1);
        cfg.interval = interval;
        MATCHED.store(0, Ordering::SeqCst);
        FIRED.store(false, Ordering::SeqCst);
        PAUSES.store(0, Ordering::SeqCst);
        for counter in CHECKPOINTS.iter().chain(ITEMS.iter()) {
            counter.store(0, Ordering::SeqCst);
        }
        MAX_GAP.store(0, Ordering::SeqCst);
        POLLS.store(0, Ordering::SeqCst);
        CANCELLED.store(0, Ordering::SeqCst);
        LINES.store(0, Ordering::SeqCst);
        STAGES.lock().unwrap_or_else(|e| e.into_inner()).clear();
        drop(cfg);
        RELEASE.fetch_add(1, Ordering::SeqCst);
    }

    /// Release a paused checkpoint (nothing if none is paused).
    pub fn release() {
        if PAUSED.load(Ordering::SeqCst) {
            RELEASE.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub(super) fn interval() -> Option<Duration> {
        config().interval
    }

    /// Record a checkpoint; Some(timeout) when it must pause.
    pub(super) fn reached(site: Site, items: usize, unit: bool) -> Option<u64> {
        CHECKPOINTS[site as usize].fetch_add(1, Ordering::SeqCst);
        ITEMS[site as usize].fetch_add(items as u64, Ordering::SeqCst);
        if unit {
            MAX_GAP.fetch_max(items as u64, Ordering::SeqCst);
        }
        let cfg = config();
        let target = cfg.stage?;
        if target.is_some_and(|target| target != site) {
            return None;
        }
        let n = MATCHED.fetch_add(1, Ordering::SeqCst) + 1;
        (n == cfg.nth && !FIRED.swap(true, Ordering::SeqCst)).then_some(cfg.timeout_ms)
    }

    /// Pause until released, cancelled or timed out; the calling thread keeps polling.
    pub(super) fn pause(
        scope: &Scope,
        word: Option<fpenv::State>,
        timeout_ms: u64,
    ) -> Result<(), EngineError> {
        let epoch = RELEASE.load(Ordering::SeqCst);
        PAUSES.fetch_add(1, Ordering::SeqCst);
        PAUSED.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let result = loop {
            if RELEASE.load(Ordering::SeqCst) != epoch {
                break Ok(());
            }
            if scope.cancelled() {
                break Err(EngineError::Cancelled);
            }
            if Instant::now() >= deadline {
                break Err(EngineError::Device(format!(
                    "testing cancel probe: pause timed out after {timeout_ms} ms; \
                     no result was published"
                )));
            }
            if let Some(word) = word {
                scope.poll_if_due(word);
            }
            std::thread::sleep(Duration::from_micros(500));
        };
        PAUSED.store(false, Ordering::SeqCst);
        result
    }

    pub(super) fn polled(attached: bool, line: bool) {
        if attached {
            POLLS.fetch_add(1, Ordering::SeqCst);
        }
        if line {
            LINES.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub(super) fn ended(cancelled: bool, stages: &[StageRow]) {
        if cancelled {
            CANCELLED.fetch_add(1, Ordering::SeqCst);
        }
        *STAGES.lock().unwrap_or_else(|e| e.into_inner()) = stages.to_vec();
    }

    pub(super) fn unit_started() {
        LIVE.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn unit_ended() {
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }

    /// A copy of the probe's statistics (§7 `cancel_probe_state`).
    #[derive(Debug, Clone)]
    pub struct Snapshot {
        pub paused: bool,
        pub pauses: u64,
        pub checkpoints: [u64; 10],
        pub items: [u64; 10],
        pub max_gap: u64,
        pub polls: u64,
        pub cancelled: u64,
        pub live_units: i64,
        pub progress_lines: u64,
        pub stages: Vec<StageRow>,
    }

    pub fn snapshot() -> Snapshot {
        let load = |counters: &[AtomicU64; 10]| {
            let mut out = [0u64; 10];
            for (slot, counter) in out.iter_mut().zip(counters) {
                *slot = counter.load(Ordering::SeqCst);
            }
            out
        };
        Snapshot {
            paused: PAUSED.load(Ordering::SeqCst),
            pauses: PAUSES.load(Ordering::SeqCst),
            checkpoints: load(&CHECKPOINTS),
            items: load(&ITEMS),
            max_gap: MAX_GAP.load(Ordering::SeqCst),
            polls: POLLS.load(Ordering::SeqCst),
            cancelled: CANCELLED.load(Ordering::SeqCst),
            live_units: LIVE.load(Ordering::SeqCst),
            progress_lines: LINES.load(Ordering::SeqCst),
            stages: STAGES.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Counts polls; cancels at the `cancel_at`-th.
    struct Count {
        polls: AtomicUsize,
        cancel_at: usize,
        lines: Mutex<Vec<String>>,
    }
    impl Watcher for Count {
        fn poll(&self, line: Option<&str>) -> PollOutcome {
            let n = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some(line) = line {
                self.lines.lock().unwrap().push(line.to_string());
            }
            PollOutcome {
                attached: true,
                cancel: n >= self.cancel_at,
                line_written: line.is_some(),
                progress_off: false,
            }
        }
    }

    fn watcher(cancel_at: usize) -> Arc<Count> {
        Arc::new(Count {
            polls: AtomicUsize::new(0),
            cancel_at,
            lines: Mutex::new(Vec::new()),
        })
    }

    #[test]
    fn outside_a_scope_checkpoints_do_nothing() {
        let mut ticker = Ticker::new(Site::Pairs);
        for _ in 0..10_000 {
            ticker.tick().unwrap();
        }
        ticker.finish().unwrap();
        assert!(current().is_none());
        assert!(check().is_ok());
    }

    #[test]
    fn wait_while_polls_during_a_wait_and_never_cuts_it_short() {
        // Outside a scope the wait returns at once without consulting `busy`.
        let mut asked = 0;
        wait_while(|| {
            asked += 1;
            true
        });
        assert_eq!(asked, 0);
        // On the calling thread it polls about every POLL_INTERVAL while busy,
        // counts no checkpoint, and returns once the flag is set.
        let count = watcher(3);
        let scope = Scope::new(count.clone(), None);
        let entered = scope.enter(fpenv::read_state());
        let start = Instant::now();
        let mut queries = 0usize;
        wait_while(|| {
            queries += 1;
            start.elapsed() < Duration::from_secs(5)
        });
        assert_eq!(count.polls.load(Ordering::SeqCst), 3);
        assert!(scope.cancelled());
        assert!(queries > 3);
        assert!(start.elapsed() >= POLL_INTERVAL * 3);
        // After the flag is set the wait returns at once (the caller's blocking
        // wait runs instead) and makes no further poll.
        let mut asked = 0;
        wait_while(|| {
            asked += 1;
            true
        });
        assert_eq!(asked, 0);
        assert_eq!(count.polls.load(Ordering::SeqCst), 3);
        drop(entered);
        scope.end();
    }

    #[test]
    fn ticker_makes_one_checkpoint_per_chunk_and_one_for_the_rest() {
        let scope = Scope::new(watcher(usize::MAX), None);
        let _entered = scope.enter(fpenv::read_state());
        stage(Stage::Pairs, 10_000);
        let mut ticker = Ticker::new(Site::Pairs);
        for _ in 0..10_000 {
            ticker.tick().unwrap();
        }
        ticker.finish().unwrap();
        // A second, batched counter reports the same items in the same chunks.
        stage(Stage::TopK1, 10_000);
        let mut batched = Ticker::new(Site::TopK1);
        for _ in 0..100 {
            batched.add(100).unwrap();
        }
        batched.finish().unwrap();
        scope.end();
        let stages = scope.stages();
        assert_eq!(stages.len(), 2);
        assert_eq!((stages[0].total, stages[0].done), (10_000, 10_000));
        assert_eq!((stages[1].total, stages[1].done), (10_000, 10_000));
        assert_eq!(stages[1].label, "topk pass 1/2");
    }

    #[test]
    fn a_cancelling_poll_stops_the_next_checkpoint_and_restores_e0() {
        let w = watcher(1);
        let scope = Scope::new(w.clone(), Some("hamming"));
        let _entered = scope.enter(fpenv::read_state());
        std::thread::sleep(POLL_INTERVAL);
        let result = boundary(Site::Pairs, 1);
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert!(scope.cancelled());
        assert_eq!(w.polls.load(Ordering::SeqCst), 1);
        // No further polls once the flag is set.
        std::thread::sleep(POLL_INTERVAL);
        assert!(boundary(Site::Pairs, 1).is_err());
        assert_eq!(w.polls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn units_on_other_threads_see_the_flag_but_never_poll() {
        let w = watcher(1);
        let scope = Scope::new(w.clone(), None);
        let _entered = scope.enter(fpenv::read_state());
        let captured = current();
        std::thread::scope(|s| {
            s.spawn(move || {
                let _unit = unit(captured);
                std::thread::sleep(POLL_INTERVAL);
                assert!(checkpoint(Site::Pairs, 1).is_ok());
            });
        });
        assert_eq!(w.polls.load(Ordering::SeqCst), 0);
        // The calling thread's timed wait polls, which cancels.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let captured = current();
        std::thread::scope(|s| {
            s.spawn(move || {
                let _unit = unit(captured);
                let _done = Done(tx);
                let mut ticker = Ticker::new(Site::Pairs);
                loop {
                    std::thread::sleep(Duration::from_millis(1));
                    if ticker.add(CHUNK).is_err() {
                        break;
                    }
                }
            });
            wait_done(&rx, 1);
        });
        assert!(scope.cancelled());
        assert!(w.polls.load(Ordering::SeqCst) >= 1);
    }

    /// Records whether the poll's code ran inside a scope.
    struct SeesScope {
        inside: Mutex<Vec<bool>>,
    }
    impl Watcher for SeesScope {
        fn poll(&self, _line: Option<&str>) -> PollOutcome {
            self.inside.lock().unwrap().push(current().is_some());
            // A nested call made by the poll's code has its own scope.
            let nested = Scope::new(watcher(usize::MAX), None);
            {
                let _entered = nested.enter(fpenv::read_state());
                assert!(Arc::ptr_eq(&current().unwrap(), &nested));
            }
            nested.end();
            self.inside.lock().unwrap().push(current().is_some());
            PollOutcome {
                attached: true,
                ..PollOutcome::default()
            }
        }
    }

    #[test]
    fn code_run_by_a_poll_sees_no_outer_scope() {
        let w = Arc::new(SeesScope {
            inside: Mutex::new(Vec::new()),
        });
        let scope = Scope::new(w.clone(), None);
        let entered = scope.enter(fpenv::read_state());
        std::thread::sleep(POLL_INTERVAL);
        assert!(boundary(Site::Pairs, 1).is_ok());
        assert_eq!(*w.inside.lock().unwrap(), vec![false, false]);
        // The outer scope is back after the poll.
        assert!(Arc::ptr_eq(&current().unwrap(), &scope));
        drop(entered);
        assert!(current().is_none());
        scope.end();
    }

    #[test]
    fn progress_lines_use_integer_formatting() {
        let w = watcher(usize::MAX);
        let scope = Scope::new(w.clone(), Some("hamming"));
        let _entered = scope.enter(fpenv::read_state());
        stage(Stage::Pairs, 10_000);
        scope.done.store(8192, Ordering::SeqCst);
        let state = scope.lock();
        let line = scope.line(&state, scope.started).unwrap();
        assert_eq!(
            line,
            "seqblaze hamming: pairs 8192/10000 pairs (81%), 0 s\n"
        );
    }
}
