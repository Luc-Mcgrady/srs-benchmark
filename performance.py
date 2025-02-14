# You can pass arguments to this script as if it were script.py
from statistics import mean
import script
import timeit
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tracemalloc

# Config
B_TIME = bool(
    os.environ.get("B", False)
)  # Runs process_wrapper_a and process_wrapper_b to compare
N = int(os.environ.get("N", 50))  # Number of users to sample
MEMORY = bool(os.environ.get("MEM", False))  # Significantly impacts run speed

# Graph Display Info
A_NAME = "orig"
B_NAME = "columns= filter="
TITLE = "filter in pq load"

# Don't change
USER_COUNT = 10000

if not MEMORY:
    noop = lambda: (0, 0)
    tracemalloc.start = noop
    tracemalloc.stop = noop
    tracemalloc.get_tracemalloc_memory = noop

sizes = []
for id in range(1, USER_COUNT):
    metadata = pq.ParquetFile(
        script.DATA_PATH / "revlogs" / f"user_id={id}" / "data.parquet"
    ).metadata

    sizes.append([id, metadata.num_rows])

sizes = sorted(sizes, key=lambda e: e[1])

indexes = range(1, USER_COUNT, USER_COUNT // N)
row_counts = [sizes[i][1] for i in indexes]

a_times = np.zeros(N)
b_times = np.zeros(N)

a_losses = np.zeros(N)
b_losses = np.zeros(N)

a_memory = np.zeros(N)
b_memory = np.zeros(N)


def process_wrapper(uid: int):
    tracemalloc.start()
    start = timeit.default_timer()
    (result, _), err = script.process(uid)
    _, memory = tracemalloc.get_traced_memory()
    time = timeit.default_timer() - start
    tracemalloc.stop()
    if err:
        print(err)
        exit(-1)
    return result, time, memory

from script import *

def create_time_series2(df):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > max_seq_len * 2].index, inplace=True)
    card_id_to_first_rating = df.groupby("card_id")["rating"].first().to_dict()
    if BINARY:
        df.loc[:, "rating"] = df.loc[:, "rating"].map({1: 1, 2: 3, 3: 3, 4: 3})
    if "delta_t" not in df.columns:
        if SECS_IVL and "elapsed_seconds" in df.columns:
            df["delta_t"] = df["elapsed_seconds"] / 86400
        elif "elapsed_days" in df.columns:
            df["delta_t"] = df["elapsed_days"]
    t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]
    df["tensor"] = [
        torch.tensor((t_item[:-1], r_item[:-1])).transpose(0, 1)
        for t_sublist, r_sublist in zip(t_history_list, r_history_list)
        for t_item, r_item in zip(t_sublist, r_sublist)
    ]
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["first_rating"] = df["card_id"].map(card_id_to_first_rating).astype(str)
    if not SECS_IVL:
        filtered_dataset = (
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered_dataset.empty:
            return pd.DataFrame()
        df[df["i"] == 2] = filtered_dataset
        df.dropna(inplace=True)
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
    if BINARY:
        df["first_rating"] = df["first_rating"].map(lambda x: "1" if x == 1 else "3")
    return df[df["elapsed_days"] > 0].sort_values(by=["review_th"])


@catch_exceptions
def process2(user_id):
    plt.close("all")
    columns = ["card_id", "day_offset", "rating", "elapsed_days"]
    if SECS_IVL:
        columns.append("elapsed_seconds")
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id), ("rating", "in", [1, 2, 3, 4])], columns=columns 
    )
    dataset = create_time_series2(df_revlogs)
    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")
    if PARTITIONS != "none":
        df_cards = pd.read_parquet(
            DATA_PATH / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            DATA_PATH / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        if PARTITIONS == "preset":
            dataset["partition"] = dataset["preset_id"].astype(int)
        elif PARTITIONS == "deck":
            dataset["partition"] = dataset["deck_id"].astype(int)
    else:
        dataset["partition"] = 0
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()
        if NO_TEST_SAME_DAY:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        testsets.append(test_set)
        partition_weights = {}
        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                if RECENCY:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)
                if DRY_RUN:
                    partition_weights[partition] = optimizer.init_w
                    continue
                if RUST:
                    train_set_items = convert_to_items(train_partition)
                    partition_weights[partition] = list(
                        map(lambda x: round(x, 4), backend.benchmark(train_set_items))
                    )
                else:
                    optimizer.define_model()
                    _ = optimizer.pretrain(dataset=train_partition, verbose=verbose)
                    if ONLY_PRETRAIN:
                        partition_weights[partition] = optimizer.init_w
                    else:
                        trainer = Trainer(
                            train_partition,
                            None,
                            optimizer.init_w,
                            n_epoch=n_epoch,
                            lr=lr,
                            gamma=gamma,
                            batch_size=batch_size,
                            max_seq_len=max_seq_len,
                            enable_short_term=not DISABLE_SHORT_TERM,
                        )
                        partition_weights[partition] = trainer.train(verbose=verbose)
            except Exception as e:
                if str(e).endswith("inadequate."):
                    if verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    tb = sys.exc_info()[2]
                    print("User:", user_id, "Error:", e.with_traceback(tb))
                partition_weights[partition] = optimizer.init_w
        w_list.append(partition_weights)

    p, y, evaluation = predict(w_list, testsets, user_id)
    last_y = y

    if PLOT:
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{path}/{user_id}.png")

    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = rmse_matrix(evaluation)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
    except:
        auc = None

    result = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
        },
        "user": user_id,
        "size": len(last_y),
        "parameters": {
            int(partition): list(map(lambda x: round(x, 6), w))
            for partition, w in w_list[-1].items()
        },
    }

    if RAW:
        raw = {
            "user": user_id,
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None

    return result, raw
import script

process1 = script.process

def process_wrapper_a(uid: int):
    script.process = process1
    return process_wrapper(uid)


def process_wrapper_b(uid: int):
    script.process = process2
    return process_wrapper(uid)


def performance_process(uid: int, i: int, wrapper, name):
    result, time, memory = wrapper(uid)
    loss = result["metrics"]["LogLoss"]
    return uid, i, time, memory, loss, name


if __name__ == "__main__":
    with ProcessPoolExecutor(script.PROCESSES) as executor:
        future_args = [
            (
                [
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_a,
                        A_NAME,
                    ),
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_b,
                        B_NAME,
                    ),
                ]
                if B_TIME
                else [
                    (
                        performance_process,
                        sizes[user_index][0],
                        i,
                        process_wrapper_a,
                        A_NAME,
                    ),
                ]
            )
            for i, user_index in enumerate(indexes)
        ]

        futures = [executor.submit(*args) for argss in future_args for args in argss]

        for future in (
            progress := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            uid, i, time, memory, loss, name = future.result()
            progress.set_description(
                f"{uid=}, rows={row_counts[i]}, {name}={time:.2f}s"
            )
            if name == A_NAME:
                a_times[i] = time
                a_losses[i] = loss
                a_memory[i] = memory
            else:
                b_times[i] = time
                b_losses[i] = loss
                b_memory[i] = memory

    total_a_time = sum(a_times)
    total_b_time = sum(b_times)

    def estimate_time(secs: int):
        return (secs * USER_COUNT) / (N * script.PROCESSES)

    print(f"total a_time for {N} users={total_a_time:.2f}s")
    if B_TIME:
        print(f"total b_time for {N} users={total_b_time:.2f}s")
    print("")

    print(
        f"Estimated total a_time ({script.PROCESSES} process)={estimate_time(total_a_time):.2f}s"
    )
    print(
        f"Estimated total a_time ({script.PROCESSES} process)={estimate_time(total_a_time) / 60 / 60:.2f}h"
    )
    if B_TIME:
        print(
            f"Estimated total b_time for {USER_COUNT} users (one process)={estimate_time(total_b_time) * USER_COUNT / N:.2f}s"
        )
        print(
            f"Estimated total b_time for {USER_COUNT} users (one process)={estimate_time(total_b_time) / 60 / 60:.2f}h"
        )

    print("")
    print(f"{mean(a_losses)=:.5f}")
    if B_TIME:
        print(f"{mean(b_losses)=:.5f}")

    GRAPHS = 2 if not MEMORY else 3

    plt.suptitle(TITLE)

    plt.subplot(1, GRAPHS, 1)
    plt.xlabel(f"Revlogs (total={sum(row_counts)})")
    plt.ylabel(f"Seconds")
    plt.plot(
        row_counts,
        a_times,
        label=f"{A_NAME} {N} in {sum(a_times):.2f}s, estimated={estimate_time(total_a_time) / 60 / 60:.2f}h",
    )
    if B_TIME:
        plt.plot(
            row_counts,
            b_times,
            label=f"{B_NAME} {N} in {sum(b_times):.2f}s, estimated={estimate_time(total_b_time) / 60 / 60:.2f}h",
        )
    plt.title(f"Time Spent")
    plt.legend()

    if MEMORY:
        plt.subplot(1, GRAPHS, 2)
        plt.xlabel(f"Revlogs")

        plt.ylabel(f"Memory (MB)")
        plt.plot(
            row_counts,
            [x / 1024 / 1024 for x in a_memory],
            label=f"{A_NAME} avg={mean(a_memory)/1024/1024:.1f}MB",
        )
        if B_TIME:
            plt.plot(
                row_counts,
                [x / 1024 / 1024 for x in b_memory],
                label=f"{B_NAME} avg={mean(b_memory)/1024/1024:.1f}MB",
            )
        plt.title(f"Memory")
        plt.legend()

    plt.subplot(1, GRAPHS, GRAPHS)
    plt.xlabel(f"Revlogs")

    plt.ylabel(f"Log Loss")
    plt.plot(row_counts, a_losses, label=f"{A_NAME} avg={mean(a_losses):.5f}")
    if B_TIME:
        plt.plot(row_counts, b_losses, label=f"{B_NAME} avg={mean(b_losses):.5f}")
    plt.title(f"Loss")
    plt.legend()

    plt.show()
