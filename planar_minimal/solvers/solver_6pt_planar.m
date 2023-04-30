function sols = solver_6pt_planar(data)
[C0,C1] = setup_elimination_template(data);
C1 = C0 \ C1;
RR = [-C1(end-2:end,:);eye(9)];
AM_ind = [9,7,1,8,2,10,11,12,3];
AM = RR(AM_ind,:);
[V,D] = eig(AM);
V = V ./ (ones(size(V,1),1)*V(1,:));
sols(1,:) = V(2,:);
sols(2,:) = diag(D).';

% Action =  y
% Quotient ring basis (V) = 1,x,x^2,x*y,x*y^2,y,y^2,y^3,y^4,
% Available monomials (RR*V) = x^2*y,x*y^3,y^5,1,x,x^2,x*y,x*y^2,y,y^2,y^3,y^4,
function [coeffs] = compute_coeffs(data)
coeffs(1) = -data(3)*data(5)*data(7) + data(2)*data(6)*data(7) + data(3)*data(4)*data(8) - data(1)*data(6)*data(8) - data(2)*data(4)*data(9) + data(1)*data(5)*data(9);
coeffs(2) = -data(6)*data(8)*data(10) + data(5)*data(9)*data(10) + data(6)*data(7)*data(11) - data(4)*data(9)*data(11) - data(5)*data(7)*data(12) + data(4)*data(8)*data(12) + data(3)*data(8)*data(13) - data(2)*data(9)*data(13) - data(3)*data(7)*data(14) + data(1)*data(9)*data(14) + data(2)*data(7)*data(15) - data(1)*data(8)*data(15) - data(3)*data(5)*data(16) + data(2)*data(6)*data(16) + data(3)*data(4)*data(17) - data(1)*data(6)*data(17) - data(2)*data(4)*data(18) + data(1)*data(5)*data(18);
coeffs(3) = -data(9)*data(11)*data(13) + data(8)*data(12)*data(13) + data(9)*data(10)*data(14) - data(7)*data(12)*data(14) - data(8)*data(10)*data(15) + data(7)*data(11)*data(15) + data(6)*data(11)*data(16) - data(5)*data(12)*data(16) - data(3)*data(14)*data(16) + data(2)*data(15)*data(16) - data(6)*data(10)*data(17) + data(4)*data(12)*data(17) + data(3)*data(13)*data(17) - data(1)*data(15)*data(17) + data(5)*data(10)*data(18) - data(4)*data(11)*data(18) - data(2)*data(13)*data(18) + data(1)*data(14)*data(18);
coeffs(4) = -data(12)*data(14)*data(16) + data(11)*data(15)*data(16) + data(12)*data(13)*data(17) - data(10)*data(15)*data(17) - data(11)*data(13)*data(18) + data(10)*data(14)*data(18);
coeffs(5) = -data(6)*data(8)*data(19) + data(5)*data(9)*data(19) + data(6)*data(7)*data(20) - data(4)*data(9)*data(20) - data(5)*data(7)*data(21) + data(4)*data(8)*data(21) + data(3)*data(8)*data(22) - data(2)*data(9)*data(22) - data(3)*data(7)*data(23) + data(1)*data(9)*data(23) + data(2)*data(7)*data(24) - data(1)*data(8)*data(24) - data(3)*data(5)*data(25) + data(2)*data(6)*data(25) + data(3)*data(4)*data(26) - data(1)*data(6)*data(26) - data(2)*data(4)*data(27) + data(1)*data(5)*data(27);
coeffs(6) = data(9)*data(14)*data(19) - data(8)*data(15)*data(19) - data(6)*data(17)*data(19) + data(5)*data(18)*data(19) - data(9)*data(13)*data(20) + data(7)*data(15)*data(20) + data(6)*data(16)*data(20) - data(4)*data(18)*data(20) + data(8)*data(13)*data(21) - data(7)*data(14)*data(21) - data(5)*data(16)*data(21) + data(4)*data(17)*data(21) - data(9)*data(11)*data(22) + data(8)*data(12)*data(22) + data(3)*data(17)*data(22) - data(2)*data(18)*data(22) + data(9)*data(10)*data(23) - data(7)*data(12)*data(23) - data(3)*data(16)*data(23) + data(1)*data(18)*data(23) - data(8)*data(10)*data(24) + data(7)*data(11)*data(24) + data(2)*data(16)*data(24) - data(1)*data(17)*data(24) + data(6)*data(11)*data(25) - data(5)*data(12)*data(25) - data(3)*data(14)*data(25) + data(2)*data(15)*data(25) - data(6)*data(10)*data(26) + data(4)*data(12)*data(26) + data(3)*data(13)*data(26) - data(1)*data(15)*data(26) + data(5)*data(10)*data(27) - data(4)*data(11)*data(27) - data(2)*data(13)*data(27) + data(1)*data(14)*data(27);
coeffs(7) = -data(15)*data(17)*data(19) + data(14)*data(18)*data(19) + data(15)*data(16)*data(20) - data(13)*data(18)*data(20) - data(14)*data(16)*data(21) + data(13)*data(17)*data(21) + data(12)*data(17)*data(22) - data(11)*data(18)*data(22) - data(12)*data(16)*data(23) + data(10)*data(18)*data(23) + data(11)*data(16)*data(24) - data(10)*data(17)*data(24) - data(12)*data(14)*data(25) + data(11)*data(15)*data(25) + data(12)*data(13)*data(26) - data(10)*data(15)*data(26) - data(11)*data(13)*data(27) + data(10)*data(14)*data(27);
coeffs(8) = -data(9)*data(20)*data(22) + data(8)*data(21)*data(22) + data(9)*data(19)*data(23) - data(7)*data(21)*data(23) - data(8)*data(19)*data(24) + data(7)*data(20)*data(24) + data(6)*data(20)*data(25) - data(5)*data(21)*data(25) - data(3)*data(23)*data(25) + data(2)*data(24)*data(25) - data(6)*data(19)*data(26) + data(4)*data(21)*data(26) + data(3)*data(22)*data(26) - data(1)*data(24)*data(26) + data(5)*data(19)*data(27) - data(4)*data(20)*data(27) - data(2)*data(22)*data(27) + data(1)*data(23)*data(27);
coeffs(9) = -data(18)*data(20)*data(22) + data(17)*data(21)*data(22) + data(18)*data(19)*data(23) - data(16)*data(21)*data(23) - data(17)*data(19)*data(24) + data(16)*data(20)*data(24) + data(15)*data(20)*data(25) - data(14)*data(21)*data(25) - data(12)*data(23)*data(25) + data(11)*data(24)*data(25) - data(15)*data(19)*data(26) + data(13)*data(21)*data(26) + data(12)*data(22)*data(26) - data(10)*data(24)*data(26) + data(14)*data(19)*data(27) - data(13)*data(20)*data(27) - data(11)*data(22)*data(27) + data(10)*data(23)*data(27);
coeffs(10) = -data(21)*data(23)*data(25) + data(20)*data(24)*data(25) + data(21)*data(22)*data(26) - data(19)*data(24)*data(26) - data(20)*data(22)*data(27) + data(19)*data(23)*data(27);
coeffs(11) = -2*data(3)^2*data(5) + 2*data(2)*data(3)*data(6) + 2*data(3)*data(4)*data(6) - 2*data(1)*data(6)^2 - 4*data(3)*data(5)*data(7) + 2*data(2)*data(6)*data(7) + 2*data(4)*data(6)*data(7) - 2*data(5)*data(7)^2 + 2*data(2)*data(3)*data(8) + 2*data(3)*data(4)*data(8) - 4*data(1)*data(6)*data(8) + 2*data(2)*data(7)*data(8) + 2*data(4)*data(7)*data(8) - 2*data(1)*data(8)^2 - 2*data(2)^2*data(9) - 4*data(2)*data(4)*data(9) - 2*data(4)^2*data(9) + 8*data(1)*data(5)*data(9);
coeffs(12) = -2*data(6)^2*data(10) - 4*data(6)*data(8)*data(10) - 2*data(8)^2*data(10) + 8*data(5)*data(9)*data(10) + 2*data(3)*data(6)*data(11) + 2*data(6)*data(7)*data(11) + 2*data(3)*data(8)*data(11) + 2*data(7)*data(8)*data(11) - 4*data(2)*data(9)*data(11) - 4*data(4)*data(9)*data(11) - 4*data(3)*data(5)*data(12) + 2*data(2)*data(6)*data(12) + 2*data(4)*data(6)*data(12) - 4*data(5)*data(7)*data(12) + 2*data(2)*data(8)*data(12) + 2*data(4)*data(8)*data(12) + 2*data(3)*data(6)*data(13) + 2*data(6)*data(7)*data(13) + 2*data(3)*data(8)*data(13) + 2*data(7)*data(8)*data(13) - 4*data(2)*data(9)*data(13) - 4*data(4)*data(9)*data(13) - 2*data(3)^2*data(14) - 4*data(3)*data(7)*data(14) - 2*data(7)^2*data(14) + 8*data(1)*data(9)*data(14) + 2*data(2)*data(3)*data(15) + 2*data(3)*data(4)*data(15) - 4*data(1)*data(6)*data(15) + 2*data(2)*data(7)*data(15) + 2*data(4)*data(7)*data(15) - 4*data(1)*data(8)*data(15) - 4*data(3)*data(5)*data(16) + 2*data(2)*data(6)*data(16) + 2*data(4)*data(6)*data(16) - 4*data(5)*data(7)*data(16) + 2*data(2)*data(8)*data(16) + 2*data(4)*data(8)*data(16) + 2*data(2)*data(3)*data(17) + 2*data(3)*data(4)*data(17) - 4*data(1)*data(6)*data(17) + 2*data(2)*data(7)*data(17) + 2*data(4)*data(7)*data(17) - 4*data(1)*data(8)*data(17) - 2*data(2)^2*data(18) - 4*data(2)*data(4)*data(18) - 2*data(4)^2*data(18) + 8*data(1)*data(5)*data(18);
coeffs(13) = -2*data(9)*data(11)^2 + 2*data(6)*data(11)*data(12) + 2*data(8)*data(11)*data(12) - 2*data(5)*data(12)^2 - 4*data(9)*data(11)*data(13) + 2*data(6)*data(12)*data(13) + 2*data(8)*data(12)*data(13) - 2*data(9)*data(13)^2 + 8*data(9)*data(10)*data(14) - 4*data(3)*data(12)*data(14) - 4*data(7)*data(12)*data(14) - 4*data(6)*data(10)*data(15) - 4*data(8)*data(10)*data(15) + 2*data(3)*data(11)*data(15) + 2*data(7)*data(11)*data(15) + 2*data(2)*data(12)*data(15) + 2*data(4)*data(12)*data(15) + 2*data(3)*data(13)*data(15) + 2*data(7)*data(13)*data(15) - 2*data(1)*data(15)^2 + 2*data(6)*data(11)*data(16) + 2*data(8)*data(11)*data(16) - 4*data(5)*data(12)*data(16) + 2*data(6)*data(13)*data(16) + 2*data(8)*data(13)*data(16) - 4*data(3)*data(14)*data(16) - 4*data(7)*data(14)*data(16) + 2*data(2)*data(15)*data(16) + 2*data(4)*data(15)*data(16) - 2*data(5)*data(16)^2 - 4*data(6)*data(10)*data(17) - 4*data(8)*data(10)*data(17) + 2*data(3)*data(11)*data(17) + 2*data(7)*data(11)*data(17) + 2*data(2)*data(12)*data(17) + 2*data(4)*data(12)*data(17) + 2*data(3)*data(13)*data(17) + 2*data(7)*data(13)*data(17) - 4*data(1)*data(15)*data(17) + 2*data(2)*data(16)*data(17) + 2*data(4)*data(16)*data(17) - 2*data(1)*data(17)^2 + 8*data(5)*data(10)*data(18) - 4*data(2)*data(11)*data(18) - 4*data(4)*data(11)*data(18) - 4*data(2)*data(13)*data(18) - 4*data(4)*data(13)*data(18) + 8*data(1)*data(14)*data(18);
coeffs(14) = -2*data(12)^2*data(14) + 2*data(11)*data(12)*data(15) + 2*data(12)*data(13)*data(15) - 2*data(10)*data(15)^2 - 4*data(12)*data(14)*data(16) + 2*data(11)*data(15)*data(16) + 2*data(13)*data(15)*data(16) - 2*data(14)*data(16)^2 + 2*data(11)*data(12)*data(17) + 2*data(12)*data(13)*data(17) - 4*data(10)*data(15)*data(17) + 2*data(11)*data(16)*data(17) + 2*data(13)*data(16)*data(17) - 2*data(10)*data(17)^2 - 2*data(11)^2*data(18) - 4*data(11)*data(13)*data(18) - 2*data(13)^2*data(18) + 8*data(10)*data(14)*data(18);
coeffs(15) = -2*data(6)^2*data(19) - 4*data(6)*data(8)*data(19) - 2*data(8)^2*data(19) + 8*data(5)*data(9)*data(19) + 2*data(3)*data(6)*data(20) + 2*data(6)*data(7)*data(20) + 2*data(3)*data(8)*data(20) + 2*data(7)*data(8)*data(20) - 4*data(2)*data(9)*data(20) - 4*data(4)*data(9)*data(20) - 4*data(3)*data(5)*data(21) + 2*data(2)*data(6)*data(21) + 2*data(4)*data(6)*data(21) - 4*data(5)*data(7)*data(21) + 2*data(2)*data(8)*data(21) + 2*data(4)*data(8)*data(21) + 2*data(3)*data(6)*data(22) + 2*data(6)*data(7)*data(22) + 2*data(3)*data(8)*data(22) + 2*data(7)*data(8)*data(22) - 4*data(2)*data(9)*data(22) - 4*data(4)*data(9)*data(22) - 2*data(3)^2*data(23) - 4*data(3)*data(7)*data(23) - 2*data(7)^2*data(23) + 8*data(1)*data(9)*data(23) + 2*data(2)*data(3)*data(24) + 2*data(3)*data(4)*data(24) - 4*data(1)*data(6)*data(24) + 2*data(2)*data(7)*data(24) + 2*data(4)*data(7)*data(24) - 4*data(1)*data(8)*data(24) - 4*data(3)*data(5)*data(25) + 2*data(2)*data(6)*data(25) + 2*data(4)*data(6)*data(25) - 4*data(5)*data(7)*data(25) + 2*data(2)*data(8)*data(25) + 2*data(4)*data(8)*data(25) + 2*data(2)*data(3)*data(26) + 2*data(3)*data(4)*data(26) - 4*data(1)*data(6)*data(26) + 2*data(2)*data(7)*data(26) + 2*data(4)*data(7)*data(26) - 4*data(1)*data(8)*data(26) - 2*data(2)^2*data(27) - 4*data(2)*data(4)*data(27) - 2*data(4)^2*data(27) + 8*data(1)*data(5)*data(27);
coeffs(16) = 8*data(9)*data(14)*data(19) - 4*data(6)*data(15)*data(19) - 4*data(8)*data(15)*data(19) - 4*data(6)*data(17)*data(19) - 4*data(8)*data(17)*data(19) + 8*data(5)*data(18)*data(19) - 4*data(9)*data(11)*data(20) + 2*data(6)*data(12)*data(20) + 2*data(8)*data(12)*data(20) - 4*data(9)*data(13)*data(20) + 2*data(3)*data(15)*data(20) + 2*data(7)*data(15)*data(20) + 2*data(6)*data(16)*data(20) + 2*data(8)*data(16)*data(20) + 2*data(3)*data(17)*data(20) + 2*data(7)*data(17)*data(20) - 4*data(2)*data(18)*data(20) - 4*data(4)*data(18)*data(20) + 2*data(6)*data(11)*data(21) + 2*data(8)*data(11)*data(21) - 4*data(5)*data(12)*data(21) + 2*data(6)*data(13)*data(21) + 2*data(8)*data(13)*data(21) - 4*data(3)*data(14)*data(21) - 4*data(7)*data(14)*data(21) + 2*data(2)*data(15)*data(21) + 2*data(4)*data(15)*data(21) - 4*data(5)*data(16)*data(21) + 2*data(2)*data(17)*data(21) + 2*data(4)*data(17)*data(21) - 4*data(9)*data(11)*data(22) + 2*data(6)*data(12)*data(22) + 2*data(8)*data(12)*data(22) - 4*data(9)*data(13)*data(22) + 2*data(3)*data(15)*data(22) + 2*data(7)*data(15)*data(22) + 2*data(6)*data(16)*data(22) + 2*data(8)*data(16)*data(22) + 2*data(3)*data(17)*data(22) + 2*data(7)*data(17)*data(22) - 4*data(2)*data(18)*data(22) - 4*data(4)*data(18)*data(22) + 8*data(9)*data(10)*data(23) - 4*data(3)*data(12)*data(23) - 4*data(7)*data(12)*data(23) - 4*data(3)*data(16)*data(23) - 4*data(7)*data(16)*data(23) + 8*data(1)*data(18)*data(23) - 4*data(6)*data(10)*data(24) - 4*data(8)*data(10)*data(24) + 2*data(3)*data(11)*data(24) + 2*data(7)*data(11)*data(24) + 2*data(2)*data(12)*data(24) + 2*data(4)*data(12)*data(24) + 2*data(3)*data(13)*data(24) + 2*data(7)*data(13)*data(24) - 4*data(1)*data(15)*data(24) + 2*data(2)*data(16)*data(24) + 2*data(4)*data(16)*data(24) - 4*data(1)*data(17)*data(24) + 2*data(6)*data(11)*data(25) + 2*data(8)*data(11)*data(25) - 4*data(5)*data(12)*data(25) + 2*data(6)*data(13)*data(25) + 2*data(8)*data(13)*data(25) - 4*data(3)*data(14)*data(25) - 4*data(7)*data(14)*data(25) + 2*data(2)*data(15)*data(25) + 2*data(4)*data(15)*data(25) - 4*data(5)*data(16)*data(25) + 2*data(2)*data(17)*data(25) + 2*data(4)*data(17)*data(25) - 4*data(6)*data(10)*data(26) - 4*data(8)*data(10)*data(26) + 2*data(3)*data(11)*data(26) + 2*data(7)*data(11)*data(26) + 2*data(2)*data(12)*data(26) + 2*data(4)*data(12)*data(26) + 2*data(3)*data(13)*data(26) + 2*data(7)*data(13)*data(26) - 4*data(1)*data(15)*data(26) + 2*data(2)*data(16)*data(26) + 2*data(4)*data(16)*data(26) - 4*data(1)*data(17)*data(26) + 8*data(5)*data(10)*data(27) - 4*data(2)*data(11)*data(27) - 4*data(4)*data(11)*data(27) - 4*data(2)*data(13)*data(27) - 4*data(4)*data(13)*data(27) + 8*data(1)*data(14)*data(27);
coeffs(17) = -2*data(15)^2*data(19) - 4*data(15)*data(17)*data(19) - 2*data(17)^2*data(19) + 8*data(14)*data(18)*data(19) + 2*data(12)*data(15)*data(20) + 2*data(15)*data(16)*data(20) + 2*data(12)*data(17)*data(20) + 2*data(16)*data(17)*data(20) - 4*data(11)*data(18)*data(20) - 4*data(13)*data(18)*data(20) - 4*data(12)*data(14)*data(21) + 2*data(11)*data(15)*data(21) + 2*data(13)*data(15)*data(21) - 4*data(14)*data(16)*data(21) + 2*data(11)*data(17)*data(21) + 2*data(13)*data(17)*data(21) + 2*data(12)*data(15)*data(22) + 2*data(15)*data(16)*data(22) + 2*data(12)*data(17)*data(22) + 2*data(16)*data(17)*data(22) - 4*data(11)*data(18)*data(22) - 4*data(13)*data(18)*data(22) - 2*data(12)^2*data(23) - 4*data(12)*data(16)*data(23) - 2*data(16)^2*data(23) + 8*data(10)*data(18)*data(23) + 2*data(11)*data(12)*data(24) + 2*data(12)*data(13)*data(24) - 4*data(10)*data(15)*data(24) + 2*data(11)*data(16)*data(24) + 2*data(13)*data(16)*data(24) - 4*data(10)*data(17)*data(24) - 4*data(12)*data(14)*data(25) + 2*data(11)*data(15)*data(25) + 2*data(13)*data(15)*data(25) - 4*data(14)*data(16)*data(25) + 2*data(11)*data(17)*data(25) + 2*data(13)*data(17)*data(25) + 2*data(11)*data(12)*data(26) + 2*data(12)*data(13)*data(26) - 4*data(10)*data(15)*data(26) + 2*data(11)*data(16)*data(26) + 2*data(13)*data(16)*data(26) - 4*data(10)*data(17)*data(26) - 2*data(11)^2*data(27) - 4*data(11)*data(13)*data(27) - 2*data(13)^2*data(27) + 8*data(10)*data(14)*data(27);
coeffs(18) = -2*data(9)*data(20)^2 + 2*data(6)*data(20)*data(21) + 2*data(8)*data(20)*data(21) - 2*data(5)*data(21)^2 - 4*data(9)*data(20)*data(22) + 2*data(6)*data(21)*data(22) + 2*data(8)*data(21)*data(22) - 2*data(9)*data(22)^2 + 8*data(9)*data(19)*data(23) - 4*data(3)*data(21)*data(23) - 4*data(7)*data(21)*data(23) - 4*data(6)*data(19)*data(24) - 4*data(8)*data(19)*data(24) + 2*data(3)*data(20)*data(24) + 2*data(7)*data(20)*data(24) + 2*data(2)*data(21)*data(24) + 2*data(4)*data(21)*data(24) + 2*data(3)*data(22)*data(24) + 2*data(7)*data(22)*data(24) - 2*data(1)*data(24)^2 + 2*data(6)*data(20)*data(25) + 2*data(8)*data(20)*data(25) - 4*data(5)*data(21)*data(25) + 2*data(6)*data(22)*data(25) + 2*data(8)*data(22)*data(25) - 4*data(3)*data(23)*data(25) - 4*data(7)*data(23)*data(25) + 2*data(2)*data(24)*data(25) + 2*data(4)*data(24)*data(25) - 2*data(5)*data(25)^2 - 4*data(6)*data(19)*data(26) - 4*data(8)*data(19)*data(26) + 2*data(3)*data(20)*data(26) + 2*data(7)*data(20)*data(26) + 2*data(2)*data(21)*data(26) + 2*data(4)*data(21)*data(26) + 2*data(3)*data(22)*data(26) + 2*data(7)*data(22)*data(26) - 4*data(1)*data(24)*data(26) + 2*data(2)*data(25)*data(26) + 2*data(4)*data(25)*data(26) - 2*data(1)*data(26)^2 + 8*data(5)*data(19)*data(27) - 4*data(2)*data(20)*data(27) - 4*data(4)*data(20)*data(27) - 4*data(2)*data(22)*data(27) - 4*data(4)*data(22)*data(27) + 8*data(1)*data(23)*data(27);
coeffs(19) = -2*data(18)*data(20)^2 + 2*data(15)*data(20)*data(21) + 2*data(17)*data(20)*data(21) - 2*data(14)*data(21)^2 - 4*data(18)*data(20)*data(22) + 2*data(15)*data(21)*data(22) + 2*data(17)*data(21)*data(22) - 2*data(18)*data(22)^2 + 8*data(18)*data(19)*data(23) - 4*data(12)*data(21)*data(23) - 4*data(16)*data(21)*data(23) - 4*data(15)*data(19)*data(24) - 4*data(17)*data(19)*data(24) + 2*data(12)*data(20)*data(24) + 2*data(16)*data(20)*data(24) + 2*data(11)*data(21)*data(24) + 2*data(13)*data(21)*data(24) + 2*data(12)*data(22)*data(24) + 2*data(16)*data(22)*data(24) - 2*data(10)*data(24)^2 + 2*data(15)*data(20)*data(25) + 2*data(17)*data(20)*data(25) - 4*data(14)*data(21)*data(25) + 2*data(15)*data(22)*data(25) + 2*data(17)*data(22)*data(25) - 4*data(12)*data(23)*data(25) - 4*data(16)*data(23)*data(25) + 2*data(11)*data(24)*data(25) + 2*data(13)*data(24)*data(25) - 2*data(14)*data(25)^2 - 4*data(15)*data(19)*data(26) - 4*data(17)*data(19)*data(26) + 2*data(12)*data(20)*data(26) + 2*data(16)*data(20)*data(26) + 2*data(11)*data(21)*data(26) + 2*data(13)*data(21)*data(26) + 2*data(12)*data(22)*data(26) + 2*data(16)*data(22)*data(26) - 4*data(10)*data(24)*data(26) + 2*data(11)*data(25)*data(26) + 2*data(13)*data(25)*data(26) - 2*data(10)*data(26)^2 + 8*data(14)*data(19)*data(27) - 4*data(11)*data(20)*data(27) - 4*data(13)*data(20)*data(27) - 4*data(11)*data(22)*data(27) - 4*data(13)*data(22)*data(27) + 8*data(10)*data(23)*data(27);
coeffs(20) = -2*data(21)^2*data(23) + 2*data(20)*data(21)*data(24) + 2*data(21)*data(22)*data(24) - 2*data(19)*data(24)^2 - 4*data(21)*data(23)*data(25) + 2*data(20)*data(24)*data(25) + 2*data(22)*data(24)*data(25) - 2*data(23)*data(25)^2 + 2*data(20)*data(21)*data(26) + 2*data(21)*data(22)*data(26) - 4*data(19)*data(24)*data(26) + 2*data(20)*data(25)*data(26) + 2*data(22)*data(25)*data(26) - 2*data(19)*data(26)^2 - 2*data(20)^2*data(27) - 4*data(20)*data(22)*data(27) - 2*data(22)^2*data(27) + 8*data(19)*data(23)*data(27);
function [C0,C1] = setup_elimination_template(data)
[coeffs] = compute_coeffs(data);
coeffs0_ind = [1,11,2,1,11,12,3,2,1,11,12,13,4,3,2,12,13,14,4,3,13,14,5,1,11,15,6,5,15,2,11,12,1,16,7,6,5,15,16,3,12,13,2,17,8,5,15,1,11,18,...
9,8,18,6,15,16,2,12,5,19,7,6,16,17,4,13,14,3,4,14];
coeffs1_ind = [10,20,10,20,8,18,10,8,18,5,15,20,10,20,9,18,19,6,16,8,9,8,18,19,7,16,17,3,13,6,20,9,19,10,10,20,19,7,17,9,9,19,17,4,14,7,7,17,14,4];
C0_ind = [1,12,13,14,17,24,25,26,27,28,29,36,37,38,39,40,41,48,50,51,52,53,61,66,68,72,73,74,77,78,79,80,83,84,85,86,87,88,89,90,91,92,95,96,97,102,104,105,106,108,...
109,110,113,114,115,116,117,118,119,120,122,123,124,125,126,127,128,131,135,136];
C1_ind = [9,10,18,20,21,22,25,30,32,33,34,36,38,41,42,43,44,45,46,47,50,51,52,53,54,55,56,57,58,59,67,69,70,71,75,76,79,81,82,83,87,88,91,93,94,95,99,100,103,107];
C0 = zeros(12,12);
C1 = zeros(12,9);
C0(C0_ind) = coeffs(coeffs0_ind);
C1(C1_ind) = coeffs(coeffs1_ind);
