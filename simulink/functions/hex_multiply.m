function q=hex_multiply(a,b)
q=[a(4)*b(1:3)+b(4)*a(1:3)+cross(a(1:3),b(1:3));a(4)*b(4)-a(1:3)'*b(1:3)];
end
