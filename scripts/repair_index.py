import os
import json

path = "data/patent_index/clip_index.json"
if not os.path.exists(path):
    print("clip_index.json does not exist!")
    exit(1)

size = os.path.getsize(path)
print(f"Original file size: {size} bytes")

# Read the last 2MB of the file
chunk_size = min(size, 2 * 1024 * 1024)
with open(path, "rb") as f:
    f.seek(size - chunk_size)
    chunk = f.read()

found = False
# Search backwards for a '}' character
for i in range(len(chunk) - 1, -1, -1):
    if chunk[i] == ord('}'):
        # Try truncating here
        truncated_chunk = chunk[:i+1]
        
        # Find the last occurrence of {"id": before this '}'
        idx_start = truncated_chunk.rfind(b'{"id":')
        if idx_start != -1:
            last_record = truncated_chunk[idx_start:]
            try:
                # Validate the last record by parsing it
                json.loads(last_record.decode('utf-8'))
                
                # If successful, this position is our truncation boundary
                truncate_pos = (size - chunk_size) + idx_start + len(last_record)
                print(f"Found valid last record at position {truncate_pos} (Offset: {idx_start})")
                
                # Truncate and seal the array with ']'
                with open(path, "r+b") as f_out:
                    f_out.seek(truncate_pos)
                    f_out.truncate()
                    f_out.write(b"]")
                
                print("Successfully repaired clip_index.json!")
                found = True
                break
            except Exception as e:
                # Not a complete record, search for the next '}' backwards
                continue

if found:
    new_size = os.path.getsize(path)
    print(f"New file size: {new_size} bytes")
    # Verify the repaired JSON is fully loadable
    try:
        print("Verifying if the whole file can be parsed...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Verification success! Parsed {len(data)} records.")
    except Exception as e:
        print(f"Verification failed: {e}")
else:
    print("Could not find any valid record boundary in the end chunk.")
