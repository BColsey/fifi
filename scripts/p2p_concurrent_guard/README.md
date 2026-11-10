# Experimental concurrent supervisor core

This is a CPU-only sequencing prototype. It has no host adapter, runnable
entry point, service integration, or permission to use GPUs. It does not
attest any actual process, endpoint, lease, scope, configuration, or service,
and it does not prove that a server-side request has drained. It cannot be used
to run a campaign.

The intended future adapter must retain the existing shared lease through
verified cleanup, and pin the secondary service identity (including process
start time, executable, unit, configuration and GPU ownership). It must contain
the companion client, prove request completion and server-side drain, prevent
unapproved service handoffs, and restore and verify the pair before release.
The adapter must enforce finite waits and keep the lease alive during manual
recovery. The core cannot establish any of those host properties itself.

An opt-in, harmless scope test checks descriptor inheritance and independent
contention on a fresh temporary lock. A future host adapter must still inspect
the inherited descriptor's inode and prove independent contention on the
canonical lock before sending any HTTP request. It must not unlock the
inherited descriptor. CPU fakes and a real, harmless scope test are both
required; a successful parent `pass_fds` call alone does not show that the
descriptor survived the service-manager scope.

The core only tests whether two supplied request time intervals overlap. This
does not prove that GPU kernels were simultaneously active, that requests
completed correctly, or that a performance measurement is uncontaminated.
Any future concurrent result must be labeled and compared against separately
measured isolated workloads.

The unit tests inject fake operations and failures. They verify sequencing and
lease-retention decisions only; they are not service, GPU, or end-to-end tests.
