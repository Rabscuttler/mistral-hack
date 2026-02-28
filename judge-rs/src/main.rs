use eframe::egui;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const APPROACHES: &[&str] = &["baseline", "prompt_engineered", "finetuned", "real"];
const TRIM_HEAD: usize = 10;
const DISPLAY_LINES: usize = 20;

#[derive(Debug, Clone, Deserialize)]
struct LyricsEntry {
    lyrics: String,
    genre: String,
    theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Judgment {
    genre: String,
    theme: String,
    left_approach: String,
    right_approach: String,
    rating: String,
    timestamp: f64,
}

#[derive(Debug, Clone)]
struct MatchPair {
    genre: String,
    theme: String,
    left_approach: String,
    right_approach: String,
}

/// A unique key for a judged pair (order-independent approaches + genre/theme).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct JudgedKey {
    genre: String,
    theme: String,
    approaches: (String, String), // sorted alphabetically
}

impl JudgedKey {
    fn new(genre: &str, theme: &str, a: &str, b: &str) -> Self {
        let approaches = if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        Self {
            genre: genre.to_string(),
            theme: theme.to_string(),
            approaches,
        }
    }
}

/// Filter mode for the sidebar.
#[derive(Debug, Clone, PartialEq)]
enum FilterMode {
    All,
    Specific(usize, usize), // indices into available
}

struct App {
    results_file: PathBuf,
    approaches: HashMap<String, Vec<LyricsEntry>>,
    available: Vec<String>,

    // All possible pairs, pre-shuffled
    all_pairs: Vec<MatchPair>,
    // Filtered view (indices into all_pairs)
    visible: Vec<usize>,
    pair_idx: usize,

    filter_mode: FilterMode,
    filter_a: usize,
    filter_b: usize,

    judgments: Vec<Judgment>,
    judged: HashSet<JudgedKey>,
    reveal: bool,
    flash: Option<(&'static str, std::time::Instant)>,
}

impl App {
    fn new(outputs_dir: PathBuf) -> Self {
        let results_file = outputs_dir.join("judgments.jsonl");
        let mut approaches = HashMap::new();
        let mut available = Vec::new();

        for &name in APPROACHES {
            let path = outputs_dir.join(format!("{name}.jsonl"));
            if let Ok(contents) = fs::read_to_string(&path) {
                let entries: Vec<LyricsEntry> = contents
                    .lines()
                    .filter_map(|l| serde_json::from_str(l).ok())
                    .collect();
                if !entries.is_empty() {
                    available.push(name.to_string());
                    approaches.insert(name.to_string(), entries);
                }
            }
        }

        let judgments = load_judgments(&results_file);

        // Build all pairs across all approach combinations
        let mut all_pairs = Vec::new();
        let n = available.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &available[i];
                let b = &available[j];
                let keys_a: HashSet<(String, String)> = approaches[a]
                    .iter()
                    .map(|e| (e.genre.clone(), e.theme.clone()))
                    .collect();
                let keys_b: HashSet<(String, String)> = approaches[b]
                    .iter()
                    .map(|e| (e.genre.clone(), e.theme.clone()))
                    .collect();
                let mut common: Vec<_> = keys_a.intersection(&keys_b).cloned().collect();
                common.sort();
                for (genre, theme) in common {
                    all_pairs.push(MatchPair {
                        genre,
                        theme,
                        left_approach: a.clone(),
                        right_approach: b.clone(),
                    });
                }
            }
        }

        // Shuffle all pairs and randomize left/right
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        all_pairs.shuffle(&mut rng);
        for pair in &mut all_pairs {
            let mut sides = vec![false, true];
            sides.shuffle(&mut rng);
            if sides[0] {
                std::mem::swap(&mut pair.left_approach, &mut pair.right_approach);
            }
        }

        // Build judged set
        let mut judged = HashSet::new();
        for j in &judgments {
            judged.insert(JudgedKey::new(
                &j.genre,
                &j.theme,
                &j.left_approach,
                &j.right_approach,
            ));
        }

        let visible: Vec<usize> = (0..all_pairs.len()).collect();

        let mut app = App {
            results_file,
            approaches,
            available,
            all_pairs,
            visible,
            pair_idx: 0,
            filter_mode: FilterMode::All,
            filter_a: 0,
            filter_b: 1,
            judgments,
            judged,
            reveal: false,
            flash: None,
        };
        app.skip_to_first_unjudged();
        app
    }

    fn rebuild_visible(&mut self) {
        self.visible = match &self.filter_mode {
            FilterMode::All => (0..self.all_pairs.len()).collect(),
            FilterMode::Specific(a, b) => {
                let name_a = &self.available[*a];
                let name_b = &self.available[*b];
                let set: HashSet<&str> = [name_a.as_str(), name_b.as_str()].into();
                (0..self.all_pairs.len())
                    .filter(|&i| {
                        let p = &self.all_pairs[i];
                        let ps: HashSet<&str> =
                            [p.left_approach.as_str(), p.right_approach.as_str()].into();
                        ps == set
                    })
                    .collect()
            }
        };
        self.pair_idx = 0;
        self.skip_to_first_unjudged();
    }

    fn skip_to_first_unjudged(&mut self) {
        for (i, &vi) in self.visible.iter().enumerate() {
            let p = &self.all_pairs[vi];
            let key = JudgedKey::new(&p.genre, &p.theme, &p.left_approach, &p.right_approach);
            if !self.judged.contains(&key) {
                self.pair_idx = i;
                return;
            }
        }
        // All judged — stay at 0
        self.pair_idx = 0;
    }

    fn current_match(&self) -> Option<&MatchPair> {
        let &vi = self.visible.get(self.pair_idx)?;
        Some(&self.all_pairs[vi])
    }

    fn current_pair(&self) -> Option<(&LyricsEntry, &LyricsEntry, &str, &str)> {
        let mp = self.current_match()?;
        let left_entry = self.approaches[&mp.left_approach]
            .iter()
            .find(|e| e.genre == mp.genre && e.theme == mp.theme)?;
        let right_entry = self.approaches[&mp.right_approach]
            .iter()
            .find(|e| e.genre == mp.genre && e.theme == mp.theme)?;
        Some((
            left_entry,
            right_entry,
            mp.left_approach.as_str(),
            mp.right_approach.as_str(),
        ))
    }

    fn submit(&mut self, rating: &str) {
        let Some(mp) = self.current_match().cloned() else {
            return;
        };
        let judgment = Judgment {
            genre: mp.genre.clone(),
            theme: mp.theme.clone(),
            left_approach: mp.left_approach.clone(),
            right_approach: mp.right_approach.clone(),
            rating: rating.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.results_file)
        {
            use std::io::Write;
            let _ = writeln!(f, "{}", serde_json::to_string(&judgment).unwrap());
        }

        self.judged.insert(JudgedKey::new(
            &mp.genre,
            &mp.theme,
            &mp.left_approach,
            &mp.right_approach,
        ));
        self.judgments.push(judgment);

        self.flash = Some((
            match rating {
                "A wins" => "Left wins!",
                "B wins" => "Right wins!",
                _ => "Tie!",
            },
            std::time::Instant::now(),
        ));

        if self.pair_idx < self.visible.len().saturating_sub(1) {
            self.pair_idx += 1;
        }
    }
}

/// Trim lyrics: skip first TRIM_HEAD lines, take next DISPLAY_LINES, drop the rest.
fn trim_lyrics(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let start = TRIM_HEAD.min(lines.len());
    let end = (start + DISPLAY_LINES).min(lines.len());
    lines[start..end].join("\n")
}

fn load_judgments(path: &PathBuf) -> Vec<Judgment> {
    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

fn main() -> eframe::Result<()> {
    let outputs_dir = PathBuf::from("outputs");
    let app = App::new(outputs_dir);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 750.0])
            .with_title("Lyrics Judge"),
        ..Default::default()
    };

    eframe::run_native("Lyrics Judge", options, Box::new(|_cc| Ok(Box::new(app))))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard input
        let mut action = None;
        ctx.input(|i| {
            if i.key_pressed(egui::Key::J) {
                action = Some("A wins");
            } else if i.key_pressed(egui::Key::K) {
                action = Some("B wins");
            } else if i.key_pressed(egui::Key::Space) {
                action = Some("Tie");
            } else if i.key_pressed(egui::Key::ArrowLeft) {
                action = Some("prev");
            } else if i.key_pressed(egui::Key::ArrowRight) {
                action = Some("next");
            }
        });
        match action {
            Some("prev") => {
                if self.pair_idx > 0 {
                    self.pair_idx -= 1;
                }
            }
            Some("next") => {
                if self.pair_idx < self.visible.len().saturating_sub(1) {
                    self.pair_idx += 1;
                }
            }
            Some(rating) => self.submit(rating),
            None => {}
        }

        // Sidebar
        egui::SidePanel::left("sidebar")
            .min_width(180.0)
            .show(ctx, |ui| {
                ui.heading("Filter");
                ui.add_space(8.0);

                // All vs specific toggle
                let mut is_all = self.filter_mode == FilterMode::All;
                if ui.checkbox(&mut is_all, "All pairs (shuffled)").changed() {
                    if is_all {
                        self.filter_mode = FilterMode::All;
                    } else {
                        self.filter_mode =
                            FilterMode::Specific(self.filter_a, self.filter_b);
                    }
                    self.rebuild_visible();
                }

                if !is_all && self.available.len() >= 2 {
                    let prev_a = self.filter_a;
                    let prev_b = self.filter_b;

                    egui::ComboBox::from_label("Approach A")
                        .selected_text(&self.available[self.filter_a])
                        .show_ui(ui, |ui| {
                            for (i, name) in self.available.iter().enumerate() {
                                ui.selectable_value(&mut self.filter_a, i, name);
                            }
                        });

                    egui::ComboBox::from_label("Approach B")
                        .selected_text(&self.available[self.filter_b])
                        .show_ui(ui, |ui| {
                            for (i, name) in self.available.iter().enumerate() {
                                if i != self.filter_a {
                                    ui.selectable_value(&mut self.filter_b, i, name);
                                }
                            }
                        });

                    if self.filter_a != prev_a || self.filter_b != prev_b {
                        if self.filter_a == self.filter_b {
                            self.filter_b =
                                (self.filter_a + 1) % self.available.len();
                        }
                        self.filter_mode =
                            FilterMode::Specific(self.filter_a, self.filter_b);
                        self.rebuild_visible();
                    }
                }

                ui.separator();
                ui.heading("Win Rates");

                let mut wins: HashMap<String, u32> = HashMap::new();
                let mut ties = 0u32;
                let total = self.judgments.len() as u32;

                for j in &self.judgments {
                    match j.rating.as_str() {
                        "A wins" => *wins.entry(j.left_approach.clone()).or_default() += 1,
                        "B wins" => *wins.entry(j.right_approach.clone()).or_default() += 1,
                        _ => ties += 1,
                    }
                }

                if total > 0 {
                    let mut sorted: Vec<_> = wins.iter().collect();
                    sorted.sort_by_key(|(k, _)| (*k).clone());
                    for (app, w) in sorted {
                        ui.label(format!(
                            "{app}: {w}/{total} ({:.0}%)",
                            100.0 * *w as f64 / total as f64
                        ));
                    }
                    if ties > 0 {
                        ui.label(format!("Ties: {ties}/{total}"));
                    }
                } else {
                    ui.label("No judgments yet.");
                }

                ui.separator();
                ui.checkbox(&mut self.reveal, "Reveal approaches");
            });

        // Main panel
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.available.len() < 2 {
                ui.heading("Need at least 2 approaches in outputs/");
                return;
            }
            if self.visible.is_empty() {
                ui.heading("No matching pairs for this filter.");
                return;
            }

            let total = self.visible.len();
            let done = self
                .visible
                .iter()
                .filter(|&&vi| {
                    let p = &self.all_pairs[vi];
                    self.judged.contains(&JudgedKey::new(
                        &p.genre,
                        &p.theme,
                        &p.left_approach,
                        &p.right_approach,
                    ))
                })
                .count();

            let mp = self.current_match().unwrap();
            let key = JudgedKey::new(
                &mp.genre,
                &mp.theme,
                &mp.left_approach,
                &mp.right_approach,
            );
            let already = if self.judged.contains(&key) {
                " (done)"
            } else {
                ""
            };
            let genre = mp.genre.clone();
            let theme = mp.theme.clone();

            ui.heading(format!(
                "{}/{total}{already}  {genre} — {theme}  [{done}/{total} judged]",
                self.pair_idx + 1,
            ));

            // Flash feedback
            if let Some((msg, when)) = &self.flash {
                if when.elapsed().as_millis() < 800 {
                    ui.colored_label(egui::Color32::from_rgb(100, 200, 100), *msg);
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                } else {
                    self.flash = None;
                }
            }

            ui.add_space(2.0);

            let Some((left, right, left_app, right_app)) = self.current_pair() else {
                return;
            };
            let left_lyrics = trim_lyrics(&left.lyrics);
            let right_lyrics = trim_lyrics(&right.lyrics);
            let left_app = left_app.to_string();
            let right_app = right_app.to_string();

            ui.small("J Left wins  |  K Right wins  |  Space Tie  |  ← → Navigate");
            ui.colored_label(
                egui::Color32::from_rgb(140, 140, 140),
                format!(
                    "Showing lines {}-{} (skipping first {TRIM_HEAD}, capped at {DISPLAY_LINES})",
                    TRIM_HEAD + 1,
                    TRIM_HEAD + DISPLAY_LINES,
                ),
            );
            ui.add_space(4.0);

            let available_width = ui.available_width();
            let half = (available_width - 20.0) / 2.0;
            let scroll_height = ui.available_height() - 4.0;

            ui.columns(2, |cols| {
                cols[0].strong("J — Left");
                if self.reveal {
                    cols[0].small(&left_app);
                }
                egui::ScrollArea::vertical()
                    .id_salt("left")
                    .max_height(scroll_height)
                    .show(&mut cols[0], |ui| {
                        ui.set_min_width(half);
                        ui.monospace(&left_lyrics);
                    });

                cols[1].strong("K — Right");
                if self.reveal {
                    cols[1].small(&right_app);
                }
                egui::ScrollArea::vertical()
                    .id_salt("right")
                    .max_height(scroll_height)
                    .show(&mut cols[1], |ui| {
                        ui.set_min_width(half);
                        ui.monospace(&right_lyrics);
                    });
            });
        });
    }
}
