OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
ry(pi/2) q[1];
cx q[1],q[0];
x q[1];
