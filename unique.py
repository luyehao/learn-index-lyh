file = open("sorted_dataset_2",'r+t',encoding='utf-8')
# file = open('sorted_demo_data','r+t',encoding='utf-8')
offset = 0
file.seek(0)
f = open('unique_time_stamp', 'w', encoding='utf-8')
pre_k = "aa"
x = []
for line in file.readlines():
    offset += len(line)
    k = line[:line.find(' ')]
    if k == pre_k:
        continue
    x.append(k)
    pre_k = k
    new_line = line
    f.write(new_line)
f.close()
print(len(x))