
def train_test_data(in_name):
    in_f = open(in_name, "r")
    train_f = open("amazon_train", "w")
    test_f = open("amazon_test", "w")
    for line in in_f:
        arr = line.strip("\n").split("\t")
        uid, mid, his_mid, clk, time = arr[0], arr[1], arr[2], arr[3], int(arr[4])
        row = clk + "\t" + uid + "\t" + mid + "\t" + his_mid + "\n"
        if time <= 1385000000:
            train_f.write(row)
        if time > 1385000000:
            test_f.write(row)

if __name__ == '__main__':
    train_test_data("remap_amazon")