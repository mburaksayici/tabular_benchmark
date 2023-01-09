# tabular_benchmark
Auto Benchmark Creator for Tabular ML Competitions


## How to Use:


```bash

tabular_benchmark = TabularBenchmark(
    datasets=["BankDataset"], models=["XGBModel", "LGBMModel"]
)
# tabular_benchmark.register_model()
# tabular_benchmark.register_dataset()

results, params = tabular_benchmark.create_benchmark()
results_df = pd.DataFrame.from_records(results)


print(
    tabulate(
        results_df,
        headers=results_df.columns,
        floatfmt=".5f",
        showindex=True,
        tablefmt="psql",
    )
)

pprint(params)
```


Results:

<img width="654" alt="Ekran Resmi 2023-01-09 17 33 13" src="https://user-images.githubusercontent.com/25187211/211332489-f470f6fa-974b-4199-ba95-dca94311288f.png">


