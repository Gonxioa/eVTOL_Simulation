function value=hex_hash_file(path)
fid=fopen(path,'rb');assert(fid>=0,'hex:MissingFile','Cannot read %s.',path);
cleanup=onCleanup(@()fclose(fid));
value=hex_sha256(fread(fid,Inf,'*uint8'));
end
