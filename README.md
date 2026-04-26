# MA-VCG-QMIX

Прототип гибридного метода управления ресурсами в edge-системе, объединяющего:

- `MA-VCG` для распределения задач по максимуму социального благосостояния с расчётом VCG-платежей;
- `QMIX` для адаптивного выбора локальных действий edge-узлов в парадигме CTDE;
- графовую модель сети, учитывающую задержки между устройствами и узлами.

Текущая реализация согласована с формализацией из диссертации:

- полезность задачи учитывает приоритет, значимость устройства, взвешенную ресурсоёмкость, задержку и затраты на передачу;
- стоимость узла учитывает вычислительные, сетевые и перегрузочные издержки с нелинейным штрафом насыщения;
- аукцион строит точное распределение для малых и средних сценариев валидации с учётом ресурсных ограничений;
- расширенные наблюдения QMIX включают аукционный контекст и маски допустимых действий;
- VCG-платежи и вклад завершённых задач интегрируются в общий командный reward.

## Структура проекта

```text
MA-VCG-QMIX/
├── README.md
├── setup.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── environment/
│   │   ├── environment.py
│   │   ├── edge_node.py
│   │   ├── task.py
│   │   └── device.py
│   ├── mechanisms/
│   │   ├── auction.py
│   │   ├── evaluation.py
│   │   └── payments.py
│   ├── agents/
│   │   ├── qmix_agent.py
│   │   ├── networks.py
│   │   └── experience_buffer.py
│   ├── learning/
│   │   ├── trainer.py
│   │   ├── benchmark.py
│   │   ├── simulator.py
│   │   └── reward_manager.py
│   └── utils/
│       ├── logger.py
│       └── visualization.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_vcg.py
│   ├── test_qmix.py
│   └── test_integration.py
├── experiments/results/
├── visualization/
│   └── plot_results.py
└── main_run_all_scenarios.py
```

## Запуск проекта

```bash
python3 -m pip install -r requirements.txt
```

Полный воспроизводимый прогон главы 6:

```bash
python3 main_run_all_scenarios.py --results-dir experiments/results/chapter6 --num-seeds 5
```

Тесты:

```bash
pytest -q
```

## Артефакты главы 6

После полного прогона автоматически формируются:

- `experiments/results/chapter6/summary_by_seed.csv` — результаты по каждому `seed`;
- `experiments/results/chapter6/summary.csv` — агрегированные метрики `mean/std/95% CI`;
- `experiments/results/chapter6/tables/*.tex` — готовые таблицы для LaTeX;
- `experiments/results/chapter6/plots/*.png` и `*.pdf` — графики для диссертации.

При необходимости можно передать `--dissertation-root /path/to/SPbU-Phd-LaTeX-Dissertation`, и тогда таблицы и рисунки будут дополнительно синхронизированы в структуру диссертации.

## Ограничения прототипа

- Точный winner determination реализован branch-and-bound поиском и потому ориентирован на исследовательские сценарии умеренного масштаба.
- Сценарии ориентированы на экспериментальную валидацию метода, а не на промышленный runtime.

## 📝 Лицензия

MIT License — свободное использование и модификация
