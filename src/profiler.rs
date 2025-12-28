use sysinfo::{Pid, System};
use tracing::info;

fn get_memory_mb() -> f64 {
    let mut system = System::new_all();
    let pid = Pid::from_u32(std::process::id());
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    system
        .process(pid)
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

pub fn with_profiler<F, R>(fname: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let mem_start = get_memory_mb();
    let time_start = std::time::Instant::now();
    let result = f();
    let time_delta = time_start.elapsed().as_secs_f64();
    let mem_delta = get_memory_mb() - mem_start;

    info!(
        fname = fname,
        mem_start = mem_start,
        mem_delta = mem_delta,
        time_delta = time_delta,
        "Profiling"
    );
    result
}
