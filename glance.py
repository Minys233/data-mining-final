import os
import random
import textwrap
import pandas as pd

df = pd.read_csv("data/train.csv")
right, total = 0, 0
while True:
    idx = random.randint(0, len(df)-1)
    print(f"序号：{idx}")
    print("新闻内容：")
    print(*textwrap.wrap(df['text'][idx], 40), sep='\n')
    print(f"\n\nPrecision: {right}/{total} = {(right/total if total > 0 else 0)*100:.2f}%")
    ans = ''
    while ans not in ['Y', 'y', 'N', 'n']:
        ans = input("Do you think its TRUE news? Y/N\n")
    if ans in ['Y', 'y']:
        ans = 0
    else:
        ans = 1
    truth = df['label'][idx]
    total += 1
    os.system('cls' if os.name == 'nt' else 'clear')
    if ans == truth:
        print(f"Right! Answer is {'N' if truth else 'Y'}")
        right += 1
    else:
        print(f"Wrong! Answer is {'N' if truth else 'Y'}")


# Record
# total entries: 38471
# true news: 19285
# fake news: 19186
# label 0 == true news, label 1 == fake news
# Precision: 270/302 = 89.40% (myself)
