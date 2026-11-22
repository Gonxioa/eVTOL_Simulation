function value=hex_sha256(bytes)
%HEX_SHA256 Digest UTF-8 text or an array of bytes.
if ischar(bytes) || isstring(bytes),bytes=unicode2native(char(bytes),'UTF-8');end
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(uint8(bytes));
value=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end
