OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
cp(1.5526789495770121) q[0],q[1];
h q[0];
