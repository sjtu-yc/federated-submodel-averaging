def write_user_data(file_name):
    central_f = open(file_name, "r")
    for line in central_f:
        arr = line.strip("\n").split("\t")
        uid = arr[1]

        user_f_name = "users/user_" + str(uid)
        user_f = open(user_f_name, "a")
        user_f.write(line)
        user_f.close()

if __name__ == '__main__':
    f_name = "../central_data/amazon_train"
    write_user_data(f_name)