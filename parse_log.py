import datetime
import time

dataset = "income_keras_sgd"

rounds = 300

real_log = []

print("time(s), accuracy, cross entropy")

# with open(f"/Users/zhiruzhu/.local/workspace/logs/{dataset}.log", "r") as log:
with open(f"./logs/{dataset}.log", "r") as log:

    lines = log.readlines()

    time_offset = None

    for line in lines:
        # print(line)
        if "METRIC" in line:
            split = line.split()
            # print(split)

            time_str = split[1]
            time_tuple = time.strptime(time_str.split(',')[0], '%H:%M:%S')
            time_in_seconds = datetime.timedelta(hours=time_tuple.tm_hour, minutes=time_tuple.tm_min,
                                                 seconds=time_tuple.tm_sec).total_seconds()
            # print(time_in_seconds)
            if time_offset is None:
                time_offset = time_in_seconds

            if "aggregator: aggregated_model_validation" in line:
            # if "aggregator: train" in line:
                accuracy = split[-2]

                print(time_in_seconds - time_offset, accuracy)
                real_log.append(str(time_in_seconds - time_offset) + ", " + accuracy + "\n")

            # if "aggregator: train" in line:
            #     cross_entropy = split[-2]
            #
            #     print(time_in_seconds - time_offset, ", ", accuracy, ", ", cross_entropy)
            #     real_log.append(str(time_in_seconds - time_offset) + ", " + accuracy + ", " + str(cross_entropy) + "\n")

# print(len(real_log))
assert len(real_log) == rounds

with open(f"./{dataset}/{dataset}.csv", "w") as f:
    f.write("time(s), accuracy\n")
    # f.write("time(s), accuracy, cross entropy\n")
    for line in real_log:
        f.write(line)

