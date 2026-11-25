//! znuts: run the No-U-Turn sampler on a built-in model.
//!
//!   znuts list
//!   znuts sample <model> [options]     draws, a summary table and diagnostics
//!   znuts trace <model> [options]      every leapfrog state of one transition
//!   znuts logp <model> [--data FILE]   log density and gradient for points on stdin

const std = @import("std");
const lib = @import("nuts");
const pdb = @import("pdb_data");
const models = lib.models;
const diag = lib.diagnostics;

const usage =
    \\usage:
    \\  znuts list
    \\  znuts sample <model> [--chains 4] [--warmup 1000] [--samples 1000] [--seed 1]
    \\                       [--delta 0.8] [--max-depth 10] [--data FILE] [--out PREFIX]
    \\                       [--save-warmup 1]
    \\  znuts trace <model> [--seed 1] [--stepsize 0.1] [--warmup 0] [--out FILE]
    \\  znuts logp <model> [--data FILE] < points.csv
    \\
    \\sample writes PREFIX-<chain>.csv (Stan's column layout) and PREFIX-summary.json
    \\when --out is given, and always prints a summary table.
    \\
;

const Options = struct {
    chains: u32 = 4,
    seed: u64 = 1,
    cfg: lib.nuts.Config = .{},
    data: ?[]const u8 = null,
    out: ?[]const u8 = null,
    stepsize: f64 = 0.1,
    trace_warmup: u32 = 0,
    save_warmup: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) fail(io, "missing command");
    const command = args[1];

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const out = &stdout.interface;

    if (std.mem.eql(u8, command, "list")) {
        inline for (models.all) |M| try out.print("{s:<28} {d:>4} parameters\n", .{ M.name, M.dim });
        try out.flush();
        return;
    }
    if (args.len < 3) fail(io, "missing model name");
    const model_name = args[2];
    const opts = parseOptions(io, args[3..]);

    inline for (models.all) |M| {
        if (std.mem.eql(u8, model_name, M.name)) {
            const json = if (opts.data) |path|
                try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited)
            else if (M.data_file) |f| pdb.get(f).? else "{}";
            const model = models.load(M, arena, json) catch |err| fail(io, @errorName(err));
            if (std.mem.eql(u8, command, "sample")) {
                try sample(M, gpa, io, &model, opts, out);
            } else if (std.mem.eql(u8, command, "trace")) {
                try trace(M, gpa, io, &model, opts, out);
            } else if (std.mem.eql(u8, command, "logp")) {
                try logp(M, gpa, io, &model, out);
            } else fail(io, "unknown command");
            try out.flush();
            return;
        }
    }
    fail(io, "unknown model (see znuts list)");
}

fn parseOptions(io: std.Io, args: []const []const u8) Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        const key = args[i];
        if (i + 1 >= args.len) fail(io, "option needs a value");
        const v = args[i + 1];
        if (std.mem.eql(u8, key, "--chains")) {
            o.chains = parseInt(io, u32, v);
        } else if (std.mem.eql(u8, key, "--warmup")) {
            o.cfg.num_warmup = parseInt(io, u32, v);
            o.trace_warmup = o.cfg.num_warmup;
        } else if (std.mem.eql(u8, key, "--samples")) {
            o.cfg.num_samples = parseInt(io, u32, v);
        } else if (std.mem.eql(u8, key, "--seed")) {
            o.seed = parseInt(io, u64, v);
        } else if (std.mem.eql(u8, key, "--delta")) {
            o.cfg.delta = std.fmt.parseFloat(f64, v) catch fail(io, "bad number");
        } else if (std.mem.eql(u8, key, "--max-depth")) {
            o.cfg.max_depth = parseInt(io, u32, v);
        } else if (std.mem.eql(u8, key, "--stepsize")) {
            o.stepsize = std.fmt.parseFloat(f64, v) catch fail(io, "bad number");
        } else if (std.mem.eql(u8, key, "--data")) {
            o.data = v;
        } else if (std.mem.eql(u8, key, "--out")) {
            o.out = v;
        } else if (std.mem.eql(u8, key, "--save-warmup")) {
            o.save_warmup = parseInt(io, u1, v) == 1;
        } else fail(io, "unknown option");
    }
    return o;
}

fn parseInt(io: std.Io, comptime T: type, s: []const u8) T {
    return std.fmt.parseInt(T, s, 10) catch fail(io, "bad integer");
}

fn chainSeed(seed: u64, chain: u64) u64 {
    return std.hash.Wyhash.hash(seed, std.mem.asBytes(&chain));
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

const ChainInfo = struct {
    warmup_seconds: f64,
    sampling_seconds: f64,
    stepsize: f64,
    divergent: u32,
    n_leapfrog: u64,
    gradient_evals: u64,
    mean_accept_stat: f64,
};

fn sample(comptime M: type, gpa: std.mem.Allocator, io: std.Io, model: *const M, opts: Options, out: *std.Io.Writer) !void {
    const P = M.out_names.len;
    const S = lib.nuts.Sampler(M);
    const n_draw = opts.cfg.num_samples;
    const n_chain = opts.chains;
    // Draws by parameter, then chain, then iteration.
    const draws = try gpa.alloc(f64, P * n_chain * n_draw);
    defer gpa.free(draws);
    const infos = try gpa.alloc(ChainInfo, n_chain);
    defer gpa.free(infos);

    for (0..n_chain) |c| {
        var s = S.init(model, chainSeed(opts.seed, c));
        s.initialize(opts.cfg.init_radius) catch fail(io, "could not find initial values with a finite density");

        var file: ?std.Io.File = null;
        var file_buf: [256 * 1024]u8 = undefined;
        var fw: std.Io.File.Writer = undefined;
        if (opts.out) |prefix| {
            var name_buf: [512]u8 = undefined;
            const path = try std.fmt.bufPrint(&name_buf, "{s}-{d}.csv", .{ prefix, c + 1 });
            file = try std.Io.Dir.cwd().createFile(io, path, .{});
            fw = file.?.writerStreaming(io, &file_buf);
            const w = &fw.interface;
            try w.print("# znuts 0.1.0\n# model = {s}\n# chain = {d}\n# seed = {d}\n", .{ M.name, c + 1, opts.seed });
            try w.writeAll("lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__");
            for (M.out_names) |nm| try w.print(",{s}", .{nm});
            try w.writeAll("\n");
        }
        defer if (file) |f| f.close(io);

        // Warmup draws go to the same file, before the adaptation summary, as in CmdStan.
        const Saver = struct {
            w: *std.Io.Writer,
            model: *const M,
            failed: bool = false,
            fn call(ctx: *anyopaque, st: lib.nuts.Stats, q: []const f64) void {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                writeRow(M, self.w, self.model, st, q[0..M.dim]) catch {
                    self.failed = true;
                };
            }
        };
        var saver: Saver = undefined;
        if (opts.save_warmup and file != null) {
            saver = .{ .w = &fw.interface, .model = model };
            s.warmup_hook = .{ .ctx = &saver, .call = Saver.call };
        }

        const t_warm = std.Io.Timestamp.now(io, .awake);
        s.warmup(opts.cfg) catch |err| fail(io, @errorName(err));
        const warm_ns = t_warm.untilNow(io, .awake).nanoseconds;

        if (file != null) {
            const w = &fw.interface;
            try w.print("# Adaptation terminated\n# Step size = {e}\n# Diagonal elements of inverse mass matrix:\n# ", .{s.epsilon});
            const im: [M.dim]f64 = s.inv_metric;
            for (im, 0..) |v, i| try w.print("{s}{e}", .{ if (i == 0) "" else ", ", v });
            try w.writeAll("\n");
        }

        var info: ChainInfo = .{ .warmup_seconds = seconds(warm_ns), .sampling_seconds = 0, .stepsize = s.epsilon, .divergent = 0, .n_leapfrog = 0, .gradient_evals = 0, .mean_accept_stat = 0 };
        const evals_before = s.gradient_evals;
        const t_samp = std.Io.Timestamp.now(io, .awake);
        for (0..n_draw) |i| {
            const st = s.transition();
            var row: [P]f64 = undefined;
            const q: [M.dim]f64 = s.z.q;
            model.write(&q, &row);
            for (row, 0..) |v, j| draws[(j * n_chain + c) * n_draw + i] = v;
            info.n_leapfrog += st.n_leapfrog;
            info.mean_accept_stat += st.accept_stat;
            if (st.divergent) info.divergent += 1;
            if (file != null) {
                const w = &fw.interface;
                try w.print("{e},{e},{e},{d},{d},{d},{e}", .{ st.lp, st.accept_stat, st.stepsize, st.treedepth, st.n_leapfrog, @intFromBool(st.divergent), st.energy });
                for (row) |v| try w.print(",{e}", .{v});
                try w.writeAll("\n");
            }
        }
        if (file != null) try fw.interface.flush();
        info.sampling_seconds = seconds(t_samp.untilNow(io, .awake).nanoseconds);
        info.gradient_evals = s.gradient_evals - evals_before;
        info.mean_accept_stat /= @floatFromInt(n_draw);
        infos[c] = info;
        if (file != null) {
            const w = &fw.interface;
            try w.print("# Elapsed Time: {d} seconds (Warm-up)\n#               {d} seconds (Sampling)\n", .{ info.warmup_seconds, info.sampling_seconds });
            try w.flush();
        }
    }

    // Diagnostics for every output.
    const sums = try gpa.alloc(diag.Summary, P);
    defer gpa.free(sums);
    for (0..P) |j| sums[j] = try diag.summarize(gpa, draws[j * n_chain * n_draw ..][0 .. n_chain * n_draw], n_chain);

    try out.print("{s}: {d} chains, {d} warmup + {d} draws each\n", .{ M.name, n_chain, opts.cfg.num_warmup, n_draw });
    try out.print("{s:<14}{s:>11}{s:>11}{s:>11}{s:>11}{s:>11}{s:>10}{s:>10}{s:>8}\n", .{ "", "mean", "sd", "5%", "50%", "95%", "ess_bulk", "ess_tail", "r_hat" });
    for (M.out_names, sums) |nm, sm| {
        try out.print("{s:<14}", .{nm});
        for ([_]f64{ sm.mean, sm.sd, sm.q5, sm.q50, sm.q95 }) |v| try printNum(out, v);
        try out.print("{d:>10.0}{d:>10.0}{d:>8.3}\n", .{ sm.ess_bulk, sm.ess_tail, sm.rhat });
    }
    var total_div: u32 = 0;
    var total_lf: u64 = 0;
    var warm_s: f64 = 0;
    var samp_s: f64 = 0;
    for (infos) |inf| {
        total_div += inf.divergent;
        total_lf += inf.n_leapfrog;
        warm_s += inf.warmup_seconds;
        samp_s += inf.sampling_seconds;
    }
    const total_draws: f64 = @floatFromInt(n_chain * n_draw);
    try out.print("divergent transitions: {d}; mean leapfrog steps per draw: {d:.2}; step sizes:", .{ total_div, @as(f64, @floatFromInt(total_lf)) / total_draws });
    for (infos) |inf| try out.print(" {d:.3}", .{inf.stepsize});
    try out.print("\ntime over all chains: warmup {d:.3} s, sampling {d:.3} s\n", .{ warm_s, samp_s });

    if (opts.out) |prefix| try writeSummary(M, io, prefix, opts, infos, sums);
}

fn writeRow(comptime M: type, w: *std.Io.Writer, model: *const M, st: lib.nuts.Stats, q: *const [M.dim]f64) !void {
    var row: [M.out_names.len]f64 = undefined;
    model.write(q, &row);
    try w.print("{e},{e},{e},{d},{d},{d},{e}", .{ st.lp, st.accept_stat, st.stepsize, st.treedepth, st.n_leapfrog, @intFromBool(st.divergent), st.energy });
    for (row) |v| try w.print(",{e}", .{v});
    try w.writeAll("\n");
}

fn printNum(out: *std.Io.Writer, v: f64) !void {
    const a = @abs(v);
    if (a != 0 and (a < 1e-3 or a >= 1e6)) return out.print("{e:>11.3}", .{v});
    if (a >= 1000) return out.print("{d:>11.1}", .{v});
    if (a >= 10) return out.print("{d:>11.3}", .{v});
    return out.print("{d:>11.4}", .{v});
}

fn writeSummary(comptime M: type, io: std.Io, prefix: []const u8, opts: Options, infos: []const ChainInfo, sums: []const diag.Summary) !void {
    var name_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&name_buf, "{s}-summary.json", .{prefix});
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writerStreaming(io, &buf);
    const w = &fw.interface;
    try w.print("{{\n  \"model\": \"{s}\",\n  \"seed\": {d},\n  \"num_warmup\": {d},\n  \"num_samples\": {d},\n  \"chains\": [\n", .{ M.name, opts.seed, opts.cfg.num_warmup, opts.cfg.num_samples });
    for (infos, 0..) |inf, i| {
        try w.print("    {{\"warmup_seconds\": {e}, \"sampling_seconds\": {e}, \"stepsize\": {e}, \"divergent\": {d}, \"n_leapfrog\": {d}, \"gradient_evals\": {d}, \"mean_accept_stat\": {e}}}{s}\n", .{ inf.warmup_seconds, inf.sampling_seconds, inf.stepsize, inf.divergent, inf.n_leapfrog, inf.gradient_evals, inf.mean_accept_stat, if (i + 1 < infos.len) "," else "" });
    }
    try w.writeAll("  ],\n  \"params\": {\n");
    for (M.out_names, sums, 0..) |nm, s, i| {
        try w.print("    \"{s}\": {{\"mean\": {e}, \"sd\": {e}, \"q5\": {e}, \"q50\": {e}, \"q95\": {e}, \"ess_bulk\": {e}, \"ess_tail\": {e}, \"rhat\": {e}}}{s}\n", .{ nm, s.mean, s.sd, s.q5, s.q50, s.q95, s.ess_bulk, s.ess_tail, s.rhat, if (i + 1 < sums.len) "," else "" });
    }
    try w.writeAll("  }\n}\n");
    try w.flush();
}

fn trace(comptime M: type, gpa: std.mem.Allocator, io: std.Io, model: *const M, opts: Options, out: *std.Io.Writer) !void {
    const S = lib.nuts.Sampler(M);
    var s = S.init(model, chainSeed(opts.seed, 0));
    s.initialize(opts.cfg.init_radius) catch fail(io, "could not initialise");
    if (opts.trace_warmup > 0) {
        var cfg = opts.cfg;
        cfg.num_warmup = opts.trace_warmup;
        s.warmup(cfg) catch |err| fail(io, @errorName(err));
    } else {
        s.epsilon = opts.stepsize;
        // Move off the initial point so the trajectory starts in the bulk.
        for (0..20) |_| _ = s.transition();
    }
    var points: std.ArrayList(lib.nuts.TracePoint) = .empty;
    defer points.deinit(gpa);
    s.trace = &points;
    s.trace_gpa = gpa;
    const st = s.transition();
    s.trace = null;

    const w = out;
    const sel: [M.dim]f64 = s.z.q;
    try w.print("{{\"model\": \"{s}\", \"stepsize\": {e}, \"treedepth\": {d}, \"n_leapfrog\": {d}, \"divergent\": {}, \"selected\": [{e}, {e}],\n \"points\": [\n", .{ M.name, s.epsilon, st.treedepth, st.n_leapfrog, st.divergent, sel[0], if (M.dim > 1) sel[1] else 0 });
    for (points.items, 0..) |p, i| {
        try w.print("  {{\"q\": [{e}, {e}], \"p\": [{e}, {e}], \"h\": {e}, \"level\": {d}, \"dir\": {d}}}{s}\n", .{ p.q[0], p.q[1], p.p[0], p.p[1], p.hamiltonian, p.level, p.direction, if (i + 1 < points.items.len) "," else "" });
    }
    try w.writeAll("]}\n");
}

fn logp(comptime M: type, gpa: std.mem.Allocator, io: std.Io, model: *const M, out: *std.Io.Writer) !void {
    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(io, &in_buf);
    const input = try stdin.interface.allocRemaining(gpa, .unlimited);
    defer gpa.free(input);
    var lines = std.mem.tokenizeAny(u8, input, "\r\n");
    while (lines.next()) |line| {
        var q: [M.dim]f64 = undefined;
        var fields = std.mem.tokenizeAny(u8, line, ", \t");
        for (&q) |*v| {
            const f = fields.next() orelse fail(io, "too few values in a row");
            v.* = std.fmt.parseFloat(f64, f) catch fail(io, "bad number");
        }
        var g: [M.dim]f64 = undefined;
        const lp = models.logDensityGrad(M, model, &q, &g);
        try out.print("{e}", .{lp});
        for (g) |v| try out.print(",{e}", .{v});
        try out.writeAll("\n");
    }
}

fn fail(io: std.Io, msg: []const u8) noreturn {
    var buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &buf);
    stderr.interface.print("znuts: {s}\n{s}", .{ msg, usage }) catch {};
    stderr.interface.flush() catch {};
    std.process.exit(1);
}
