# P2P pair host observer: experimental read-only slice

`LinuxPairObserver` samples the running pair service without starting or
stopping it. It pins the loaded unit and process identity, fingerprints
explicitly supplied launcher/configuration/environment/key and unit files,
checks the process `--config` argument, and compares two snapshots around
three idle GPU samples. It refuses active client sockets and requires the
pair service process to own the reported listening socket and all reported
compute-app contexts on the specified pair. Its test inputs are synthetic;
there is no executable entry point or live admission integration.

This slice **cannot authorize a service stop or a concurrent campaign**. The
caller must prove the canonical held lease, engine readiness and request queue,
all GPU contexts including graphics, companion and scope identity, drain,
restoration inputs and lifecycle containment. Fingerprinted environment and
key paths are supplied by the caller and are not proof that the launcher reads
those files. The observer does not enumerate model or restoration dependencies.
Samples do not fence future requests or direct same-user service changes, and
the code intentionally refuses router-poll connections. Because a snapshot
repeatedly hashes the running executable and waits between utilization samples,
it is not a per-second performance monitor.

An independent read-only check found that this host's `systemctl show --all`
omits several empty hook/environment properties, while direct systemd D-Bus
queries report typed empty arrays. For those known omitted fields the observer
resolves the unit with `GetUnit` and requires the exact expected empty D-Bus
signature; it rejects nonempty, unsupported, missing or changing proofs. This
does not treat missing fields as empty. No live observer, service handoff, GPU
workload or 2080 integration has been run. CPU-only fake-command/proc tests do
not qualify live behavior.
