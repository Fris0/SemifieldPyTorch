limit = 128000000 # 128 MB in bytes

# Initialize
bytes_list = []
current = 64000
step = 64000
step_count = 0

# Fill the list
while current < limit:
    bytes_list.append(current)
    current += step
    step_count += 1
    if step_count == 4:
        step_count = 0
        step = current

print(bytes_list)