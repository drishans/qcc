OPENQASM 3.0;
include "stdgates.inc";
// A circuit that is mostly air: everything but the Bell pair cancels.
qubit[2] q;
bit[2] c;
h q[0];
x q[1];
x q[1];
h q[0];
h q[0];
t q[0];
tdg q[0];
cx q[0], q[1];
cx q[0], q[1];
rz(0.7) q[1];
rz(-0.7) q[1];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
