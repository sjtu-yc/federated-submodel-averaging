
def stat_process_data(file_name):
    f = open(file_name, "r")
    train_cnt = 0.0
    ttl_cnt = 0.0
    for line in f:
        arr = line.strip("\n").split("\t")
        time = int(arr[4])
        ttl_cnt += 1
        if time <= 1385000000:
            train_cnt += 1
    print(train_cnt / ttl_cnt)

def stat_uid_mid(file_name):
    uids, mids = set(), set()
    f = open(file_name, "r")
    for line in f:
        arr = line.strip("\n").split("\t")
        uid = int(arr[0])
        mid = int(arr[1])
        uids.add(uid)
        mids.add(mid)
    print(len(uids), len(mids))
    return uids, mids

if __name__ == '__main__':
    stat_process_data("remap_amazon")