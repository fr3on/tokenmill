use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, List, ListItem, ListState, Paragraph, Row, Sparkline, Table, Tabs},
    Terminal,
};
use std::sync::mpsc::{self, Receiver};
use sysinfo::System;
use tokenmill_core::ProgressUpdate;

// ─── Dracula palette ─────────────────────────────────────────────────────────
const PURPLE: Color = Color::Rgb(189, 147, 249);
const CYAN:   Color = Color::Rgb(139, 233, 253);
const GREEN:  Color = Color::Rgb(80,  250, 123);
const YELLOW: Color = Color::Rgb(241, 250, 140);
const PINK:   Color = Color::Rgb(255, 121, 198);
const BG:     Color = Color::Rgb(40,  42,  54);
const SEL:    Color = Color::Rgb(68,  71,  90);
const RED:    Color = Color::Rgb(255, 85,  85);

// ─── Internal event ──────────────────────────────────────────────────────────
enum Event {
    Input(CEvent),
    Tick,
    Progress(ProgressUpdate),
}

// ─── Tab ─────────────────────────────────────────────────────────────────────
#[derive(Default, PartialEq, Clone, Copy)]
pub enum Tab { #[default] Dashboard, Logs, Help }

impl Tab {
    fn next(self) -> Self { match self { Tab::Dashboard => Tab::Logs, Tab::Logs => Tab::Help, Tab::Help => Tab::Dashboard } }
    fn prev(self) -> Self { match self { Tab::Dashboard => Tab::Help, Tab::Logs => Tab::Dashboard, Tab::Help => Tab::Logs } }
    fn index(self) -> usize { match self { Tab::Dashboard => 0, Tab::Logs => 1, Tab::Help => 2 } }
}

// ─── App state ────────────────────────────────────────────────────────────────
pub struct App {
    pub progress:           ProgressUpdate,
    pub start_time:         Instant,
    pub done:               bool,
    throughput_history:     VecDeque<u64>,  // lines/tick  (capped 120 = 12 s)
    bytes_history:          VecDeque<u64>,  // bytes/tick
    prev_lines:             u64,
    prev_bytes:             u64,
    pub total_bytes:        Option<u64>,    // real file size → accurate %
    pub active_tab:         Tab,
    pub logs:               VecDeque<String>,
    pub log_state:          ListState,
    pub output:             Option<String>, // result payload (e.g. stats JSON)
    sys:                    System,
    pub cpu_usage:          f32,
    pub mem_used_mb:        u64,
    pub mem_total_mb:       u64,
    pub lines_per_sec:      f64,
    pub bytes_per_sec:      f64,
    pub error_count:        u64,
}

impl App {
    pub fn new(total_bytes: Option<u64>) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        let mut log_state = ListState::default();
        log_state.select(Some(0));
        App {
            progress: ProgressUpdate::default(),
            start_time: Instant::now(),
            done: false,
            throughput_history: VecDeque::with_capacity(120),
            bytes_history:      VecDeque::with_capacity(120),
            prev_lines: 0, prev_bytes: 0,
            total_bytes,
            active_tab: Tab::default(),
            logs: { let mut d = VecDeque::with_capacity(1000); d.push_back("[SYSTEM] Dashboard initialised.".to_string()); d },
            log_state,
            output: None,
            sys,
            cpu_usage: 0.0, mem_used_mb: 0, mem_total_mb: 0,
            lines_per_sec: 0.0, bytes_per_sec: 0.0,
            error_count: 0,
        }
    }

    pub fn on_tick(&mut self) {
        self.sys.refresh_cpu();
        self.sys.refresh_memory();
        let cpus = self.sys.cpus();
        self.cpu_usage = if cpus.is_empty() { 0.0 } else { cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32 };
        self.mem_total_mb = self.sys.total_memory() / 1_048_576;
        self.mem_used_mb  = self.sys.used_memory()  / 1_048_576;

        let line_delta = self.progress.lines_processed.saturating_sub(self.prev_lines);
        self.prev_lines = self.progress.lines_processed;
        if self.throughput_history.len() >= 120 { self.throughput_history.pop_front(); }
        self.throughput_history.push_back(line_delta);

        let byte_delta = self.progress.bytes_processed.saturating_sub(self.prev_bytes);
        self.prev_bytes = self.progress.bytes_processed;
        if self.bytes_history.len() >= 120 { self.bytes_history.pop_front(); }
        self.bytes_history.push_back(byte_delta);

        // Smooth over the last ~1 s (10 ticks × 100 ms)
        let n = 10_usize.min(self.throughput_history.len());
        if n > 0 {
            self.lines_per_sec = self.throughput_history.iter().rev().take(n).sum::<u64>() as f64 / (n as f64 * 0.1);
            self.bytes_per_sec  = self.bytes_history.iter().rev().take(n).sum::<u64>() as f64 / (n as f64 * 0.1);
        }
    }

    pub fn on_progress(&mut self, update: ProgressUpdate) {
        if update.status.contains("error") || update.status.contains("ERR") { self.error_count += 1; }
        if update.lines_processed > 0 && update.lines_processed.is_multiple_of(10_000) {
            self.push_log(format!("[{:.1}s] {} lines · {}", self.start_time.elapsed().as_secs_f32(), fmt_number(update.lines_processed), fmt_bytes(update.bytes_processed)));
        }
        if update.status == "Finished" {
            self.done = true;
            if let Some(ref o) = update.output { self.output = Some(o.clone()); }
            self.push_log(format!("[{:.1}s] ✓ DONE — {} lines  {}", self.start_time.elapsed().as_secs_f32(), fmt_number(update.lines_processed), fmt_bytes(update.bytes_processed)));
        }
        self.progress = update;
    }

    fn push_log(&mut self, msg: String) {
        if self.logs.len() >= 1000 { self.logs.pop_front(); }
        self.logs.push_back(msg);
    }

    pub fn scroll_down(&mut self) {
        let max = self.logs.len().saturating_sub(1);
        let i = self.log_state.selected().map(|i| (i + 1).min(max)).unwrap_or(0);
        self.log_state.select(Some(i));
    }
    pub fn scroll_up(&mut self) {
        let i = self.log_state.selected().map(|i| i.saturating_sub(1)).unwrap_or(0);
        self.log_state.select(Some(i));
    }
}

// ─── Terminal restore helper ──────────────────────────────────────────────────
fn restore_terminal(terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>) {
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture);
    let _ = terminal.show_cursor();
}

// ─── Public entry-point ───────────────────────────────────────────────────────
/// Runs the TUI as a standalone system-monitor dashboard (no worker task).
pub fn run_tui_standalone() -> anyhow::Result<()> {
    // Send one seed update to set a nice initial status, then let the sender
    // drop so the bridge thread exits immediately. The TUI keeps running on
    // ticks and keyboard events.
    let (tx, rx) = std::sync::mpsc::channel::<ProgressUpdate>();
    let _ = tx.send(ProgressUpdate {
        status: "System Monitor".to_string(),
        ..Default::default()
    });
    drop(tx);
    run_tui(rx, None)?;
    Ok(())
}

/// Runs the TUI, consuming progress updates from `rx`.
/// Returns `Ok(Some(payload))` when the worker attached a result (e.g. stats JSON).
pub fn run_tui(rx: Receiver<ProgressUpdate>, total_bytes: Option<u64>) -> anyhow::Result<Option<String>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Panic hook: restore terminal before printing the panic message.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen, DisableMouseCapture);
        original_hook(info);
    }));

    let mut app = App::new(total_bytes);
    let (tx, event_rx) = mpsc::channel::<Event>();
    let tick_rate = Duration::from_millis(100);

    // Input + tick thread
    let tx_input = tx.clone();
    std::thread::spawn(move || {
        let mut last_tick = Instant::now();
        loop {
            let timeout = tick_rate.checked_sub(last_tick.elapsed()).unwrap_or(Duration::ZERO);
            if event::poll(timeout).unwrap_or(false) {
                if let Ok(ev) = event::read() {
                    if tx_input.send(Event::Input(ev)).is_err() { break; }
                }
            }
            if last_tick.elapsed() >= tick_rate {
                if tx_input.send(Event::Tick).is_err() { break; }
                last_tick = Instant::now();
            }
        }
    });

    // Progress bridge thread
    let tx_prog = tx;
    std::thread::spawn(move || {
        while let Ok(update) = rx.recv() {
            if tx_prog.send(Event::Progress(update)).is_err() { break; }
        }
        // Send one last tick so the loop can observe the done state.
        let _ = tx_prog.send(Event::Tick);
    });

    let result = run_loop(&mut terminal, &mut app, event_rx);
    restore_terminal(&mut terminal);
    result.map_err(|e| anyhow::anyhow!(e))?;
    Ok(app.output.take())
}

// ─── Event loop ──────────────────────────────────────────────────────────────
fn run_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut App, rx: Receiver<Event>) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        let ev = match rx.recv() {
            Ok(e) => e,
            Err(_) => return Ok(()),   // all senders dropped
        };

        match ev {
            Event::Tick => app.on_tick(),
            Event::Progress(u) => app.on_progress(u),
            Event::Input(cev) => {
                if let CEvent::Key(key) = cev {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
                        KeyCode::Enter if app.done => return Ok(()),
                        KeyCode::Char('1') => app.active_tab = Tab::Dashboard,
                        KeyCode::Char('2') => app.active_tab = Tab::Logs,
                        KeyCode::Char('3') => app.active_tab = Tab::Help,
                        KeyCode::Tab     => app.active_tab = app.active_tab.next(),
                        KeyCode::BackTab => app.active_tab = app.active_tab.prev(),
                        KeyCode::Right   => app.active_tab = app.active_tab.next(),
                        KeyCode::Left    => app.active_tab = app.active_tab.prev(),
                        KeyCode::Down    => app.scroll_down(),
                        KeyCode::Up      => app.scroll_up(),
                        KeyCode::PageDown => (0..10).for_each(|_| app.scroll_down()),
                        KeyCode::PageUp   => (0..10).for_each(|_| app.scroll_up()),
                        _ => {}
                    }
                }
            }
        }
    }
}

// ─── Top-level layout ─────────────────────────────────────────────────────────
fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(3), Constraint::Min(0), Constraint::Length(1)])
        .split(f.size());

    render_health_header(f, app, chunks[0]);
    render_tabs(f, app, chunks[1]);
    match app.active_tab {
        Tab::Dashboard => render_dashboard(f, app, chunks[2]),
        Tab::Logs      => render_logs(f, app, chunks[2]),
        Tab::Help      => render_help(f, chunks[2]),
    }
    render_status_bar(f, app, chunks[3]);
}

fn render_health_header(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mem_pct = if app.mem_total_mb > 0 { app.mem_used_mb * 100 / app.mem_total_mb } else { 0 };
    let cpu_color = if app.cpu_usage > 80.0 { RED } else if app.cpu_usage > 50.0 { YELLOW } else { CYAN };
    let mem_color = if mem_pct > 80 { RED } else if mem_pct > 60 { YELLOW } else { PINK };
    let line = Line::from(vec![
        Span::raw("  CPU ").bold(),
        Span::styled(format!("{:5.1}%", app.cpu_usage), Style::default().fg(cpu_color)),
        Span::raw("  │  RAM ").fg(Color::DarkGray),
        Span::styled(format!("{}/{} MB  {}%", app.mem_used_mb, app.mem_total_mb, mem_pct), Style::default().fg(mem_color)),
        Span::raw("  │  ").fg(Color::DarkGray),
        Span::styled("tokenmill", Style::default().fg(PURPLE).add_modifier(Modifier::BOLD)),
        Span::raw("  Streaming Pipeline").fg(Color::DarkGray),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_tabs(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let tabs = Tabs::new(vec!["[1] MONITOR", "[2] ACTIVITY", "[3] HELP"])
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(SEL)))
        .select(app.active_tab.index())
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(Style::default().fg(PURPLE).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, area);
}

fn render_dashboard(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    let elapsed = app.start_time.elapsed().as_secs_f64();
    let eta_str = if app.done {
        "done".to_string()
    } else if let Some(total) = app.total_bytes {
        if app.bytes_per_sec > 1.0 {
            fmt_duration(total.saturating_sub(app.progress.bytes_processed) as f64 / app.bytes_per_sec)
        } else { "--".to_string() }
    } else { "--".to_string() };

    let rows = vec![
        Row::new(vec![Cell::from("  LINES  ").fg(Color::Gray), Cell::from(fmt_number(app.progress.lines_processed)).fg(CYAN).bold()]),
        Row::new(vec![Cell::from("  LINES/s").fg(Color::Gray), Cell::from(format!("{:.0}", app.lines_per_sec)).fg(GREEN).bold()]),
        Row::new(vec![Cell::from("  BYTES  ").fg(Color::Gray), Cell::from(fmt_bytes(app.progress.bytes_processed)).fg(CYAN)]),
        Row::new(vec![Cell::from("  BYTES/s").fg(Color::Gray), Cell::from(format!("{}/s", fmt_bytes(app.bytes_per_sec as u64))).fg(YELLOW)]),
        Row::new(vec![Cell::from("  ETA    ").fg(Color::Gray), Cell::from(eta_str).fg(if app.done { GREEN } else { YELLOW })]),
        Row::new(vec![Cell::from("  ELAPSED").fg(Color::Gray), Cell::from(fmt_duration(elapsed)).fg(Color::Gray)]),
        Row::new(vec![Cell::from("  ERRORS ").fg(Color::Gray), Cell::from(format!("{}", app.error_count)).fg(if app.error_count > 0 { RED } else { GREEN })]),
    ];
    let table = Table::new(rows, [Constraint::Length(11), Constraint::Min(0)])
        .block(Block::default().borders(Borders::ALL).title(" TASK METRICS ").border_style(Style::default().fg(SEL)))
        .style(Style::default().fg(Color::White));
    f.render_widget(table, top[0]);

    let spark_data: Vec<u64> = app.throughput_history.iter().copied().collect();
    let (stitle, scolor) = if app.done { (" THROUGHPUT (complete) ", Color::DarkGray) } else { (" THROUGHPUT (lines/tick) ", PINK) };
    let sparkline = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(stitle).border_style(Style::default().fg(SEL)))
        .data(&spark_data)
        .style(Style::default().fg(scolor));
    f.render_widget(sparkline, top[1]);

    // Mini log (newest first)
    let log_h = chunks[1].height.saturating_sub(2) as usize;
    let mini_items: Vec<ListItem> = app.logs.iter().rev().take(log_h)
        .map(|l| ListItem::new(Span::raw(l).fg(log_color(l)))).collect();
    f.render_widget(
        List::new(mini_items).block(Block::default().borders(Borders::ALL).title(" RECENT ACTIVITY ").border_style(Style::default().fg(SEL))),
        chunks[1],
    );

    // Gauge
    let pct = gauge_pct(app);
    let gstyle = if app.done { Style::default().fg(GREEN).bg(BG).add_modifier(Modifier::BOLD) }
                 else        { Style::default().fg(PURPLE).bg(BG).add_modifier(Modifier::BOLD) };
    let label = if app.done { format!("COMPLETE  {}", fmt_bytes(app.progress.bytes_processed)) }
                else if app.total_bytes.is_some() { format!("{}%", pct) }
                else { format!("{} processed", fmt_bytes(app.progress.bytes_processed)) };
    f.render_widget(
        Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" PROGRESS ").border_style(Style::default().fg(SEL)))
            .gauge_style(gstyle).percent(pct).label(label),
        chunks[2],
    );
}

fn render_logs(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app.logs.iter()
        .map(|l| ListItem::new(Span::raw(l).fg(log_color(l)))).collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" ACTIVITY LOG   ↑↓ PgUp·PgDn to scroll ").border_style(Style::default().fg(SEL)))
        .highlight_style(Style::default().bg(SEL).fg(CYAN))
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, area, &mut app.log_state);
}

fn render_help(f: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(9)])
        .split(area);

    let cmd_rows = vec![
        Row::new(vec!["validate",    "--schema <fmt>",                    "Check schema conformance"]),
        Row::new(vec!["stats",       "--tokenizer",                        "Token & length distribution"]),
        Row::new(vec!["filter",      "--min/max-tokens / --perplexity",    "Filter by length or perplexity"]),
        Row::new(vec!["convert",     "--from --to",                        "Format conversion"]),
        Row::new(vec!["dedup",       "--method exact|minhash|semantic",    "Remove duplicates"]),
        Row::new(vec!["sample",      "--n --seed",                         "Reservoir sampling"]),
        Row::new(vec!["hf list",     "<repo_id> [--type]",                 "List files in HF repo"]),
        Row::new(vec!["hf download", "<repo_id> <file> [--type]",          "Download from HF Hub"]),
        Row::new(vec!["(any)",       "--tui",                              "Show this dashboard"]),
        Row::new(vec!["(any)",       "--quiet",                            "Suppress progress"]),
    ];
    let cmd_table = Table::new(cmd_rows, [Constraint::Length(14), Constraint::Length(34), Constraint::Min(0)])
        .header(Row::new(vec!["COMMAND", "FLAGS", "DESCRIPTION"])
            .style(Style::default().fg(PURPLE).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)))
        .block(Block::default().borders(Borders::ALL).title(" COMMAND REFERENCE ").border_style(Style::default().fg(SEL)))
        .column_spacing(2);
    f.render_widget(cmd_table, chunks[0]);

    let key_rows = vec![
        Row::new(vec!["q / Esc",      "Quit at any time"]),
        Row::new(vec!["Enter",        "Exit after task finishes"]),
        Row::new(vec!["1 / 2 / 3",   "Switch tabs directly"]),
        Row::new(vec!["Tab / ← →",   "Cycle tabs"]),
        Row::new(vec!["↑ ↓ PgUp Dn", "Scroll log (Activity tab)"]),
        Row::new(vec!["Ctrl-C",       "Force quit"]),
    ];
    let key_table = Table::new(key_rows, [Constraint::Length(16), Constraint::Min(0)])
        .header(Row::new(vec!["KEY", "ACTION"])
            .style(Style::default().fg(CYAN).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)))
        .block(Block::default().borders(Borders::ALL).title(" KEYBINDINGS ").border_style(Style::default().fg(SEL)))
        .column_spacing(2);
    f.render_widget(key_table, chunks[1]);
}

fn render_status_bar(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let (mode_txt, mode_bg) = if app.done { (" ✓ FINISHED ", GREEN) } else { ("  RUNNING  ", YELLOW) };
    let prompt = if app.done { "  ↵ Enter or q to exit" } else { "  q Quit  Tab Tabs  ↑↓ Scroll" };
    let text = Line::from(vec![
        Span::styled(mode_txt, Style::default().bg(mode_bg).fg(Color::Black).bold()),
        Span::raw("  │  ").fg(Color::DarkGray),
        Span::raw(app.progress.status.as_str()).fg(CYAN).bold(),
        Span::raw(prompt).fg(Color::DarkGray),
        Span::raw("  │  ").fg(Color::DarkGray),
        Span::raw(fmt_duration(app.start_time.elapsed().as_secs_f64()) + " elapsed").fg(Color::DarkGray),
    ]);
    f.render_widget(Paragraph::new(text).style(Style::default().bg(BG)), area);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
fn gauge_pct(app: &App) -> u16 {
    if app.done { return 100; }
    match app.total_bytes {
        Some(t) if t > 0 => ((app.progress.bytes_processed * 100) / t).min(100) as u16,
        _ => 0,
    }
}

fn log_color(line: &str) -> Color {
    if line.contains('✓') || line.contains("DONE") { GREEN }
    else if line.contains("ERR") || line.contains("error") { RED }
    else if line.contains("SYSTEM") { PURPLE }
    else { Color::Gray }
}

pub fn fmt_bytes(b: u64) -> String {
    match b {
        0..=1_023             => format!("{} B",   b),
        1_024..=1_048_575     => format!("{:.1} KB", b as f64 / 1_024.0),
        1_048_576..=1_073_741_823 => format!("{:.1} MB", b as f64 / 1_048_576.0),
        _                     => format!("{:.2} GB", b as f64 / 1_073_741_824.0),
    }
}

fn fmt_number(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn fmt_duration(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 { format!("{}s", s) }
    else if s < 3600 { format!("{}m{}s", s / 60, s % 60) }
    else { format!("{}h{}m", s / 3600, (s % 3600) / 60) }
}
