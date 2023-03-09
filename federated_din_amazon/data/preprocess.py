from random import choice

def read_user_data(file_name):
    user_data = {}
    item_set = set()
    f = open(file_name, "r")
    for line in f:
        arr = line.strip("\n").split(",")
        uid, mid, clk, time = arr[0], arr[1], arr[2], arr[3]
        item_set.add(mid)

        if float(clk) >= 5: 
            clk = 1
        else:
            clk = 0

        if uid not in user_data:
            user_data[uid] = {}
            while time in user_data[uid]:
                time = str(int(time)+1)
            user_data[uid][time] = ((mid, clk))
        else:
            while time in user_data[uid]:
                time = str(int(time)+1)
            user_data[uid][time] = ((mid, clk))

    return user_data, item_set
    
def preprocess_user_data(user_data, item_set, out_file):
    out_f = open(out_file, "w")
    for uid in user_data:
        uid_data = user_data[uid]
        if len(uid_data) < 40:
            continue
        his_mid = ""
        his_mid_set = set()
        for time in sorted(uid_data):
            mid, clk = uid_data[time]
            line = str(uid) + "\t" + str(mid) + "\t" + his_mid + "\t" + str(clk) + "\t" + str(time) + "\n"
            out_f.write(line)
            if clk == 1:
                his_mid_set.add(mid)
                if his_mid != "":
                    his_mid = his_mid + "," + mid
                else:
                    his_mid = mid

            # for _ in range(5):
            #     item_idx = choice(item_set)
            #     if item_idx not in his_mid_set:
            #         line = str(uid) + "\t" + str(item_idx) + "\t" + his_mid + "\t" + str(0) + "\t" + str(time) + "\n"
            #         out_f.write(line)
            
def remap_id(in_file):
    in_f = open(in_file, "r")
    uid_set, mid_set = set(), set()
    for line in in_f:
        arr = line.strip("\n").split("\t")
        uid, mid = arr[0], arr[1]
        uid_set.add(uid)
        mid_set.add(mid)
    in_f.close()

    uid_remap, mid_remap = {}, {}
    for (i, uid) in enumerate(sorted(uid_set)):
        uid_remap[uid] = i+1
    for (j, mid) in enumerate(sorted(mid_set)):
        mid_remap[mid] = j+1

    uid_remap[''] = 0
    mid_remap[''] = 0
    return uid_remap, mid_remap

def write_newid_file(uid_remap, mid_remap, in_file, out_file):
    in_f = open(in_file, "r")
    out_f = open(out_file, "w")
    for line in in_f:
        arr = line.strip("\n").split("\t")
        uid, mid, his_mid, clk, time = arr[0], arr[1], arr[2], arr[3], arr[4]
        uuid = uid_remap[uid]
        mmid = mid_remap[mid]

        mids = his_mid.split(",")
        his_mmids = str(mid_remap[mids[0]])
        for i in range(len(mids)-1):
            his_mmids = his_mmids + "," + str(mid_remap[mids[i+1]])

        row = str(uuid) + "\t" + str(mmid) + "\t" + his_mmids + "\t" + str(clk) + "\t" + str(time) + "\n"
        out_f.write(row)
    
    in_f.close()
    out_f.close()

if __name__ == '__main__':
    user_data, item_set = read_user_data("rating_only.csv")
    preprocess_user_data(user_data, list(item_set), "data_5")
    uid_remap, mid_remap = remap_id("data_5")
    print(len(uid_remap), len(mid_remap))
    write_newid_file(uid_remap, mid_remap, "data_5", "remap_amazon")