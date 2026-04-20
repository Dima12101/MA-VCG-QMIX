# MA-VCG-QMIX

Прототип гибридного метода управления ресурсами в edge-системе, объединяющего:

- `MA-VCG` для распределения задач по максимуму социального благосостояния с расчётом VCG-платежей;
- `QMIX` для адаптивного выбора локальных действий edge-узлов в парадигме CTDE;
- графовую модель сети, учитывающую задержки между устройствами и узлами.

Текущая реализация согласована с формализацией из диссертации:

- полезность задачи учитывает приоритет, значимость устройства, задержку и затраты на передачу;
- стоимость узла учитывает вычислительные, сетевые и перегрузочные издержки;
- аукцион строит глобальное распределение с учётом ресурсных ограничений;
- VCG-платежи интегрируются в функцию вознаграждения QMIX.

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
│   │   ├── simulator.py
│   │   └── reward_manager.py
│   └── utils/
│       ├── logger.py
│       └── visualization.py
├── tests/
│   ├── test_vcg.py
│   ├── test_qmix.py
│   └── test_integration.py
├── experiments/
│   ├── scenario_1_baseline.py
│   ├── scenario_2_high_load.py
│   ├── scenario_3_heterogeneous.py
│   └── scenario_4_dynamic.py
├── visualization/
│   └── plot_results.py
└── main_run_all_scenarios.py
```

## Запуск проекта

```bash
pip install -r requirements.txt
python experiments/scenario_1_baseline.py
python visualization/plot_results.py
```

Полный прогон всех сценариев:

```bash
python main_run_all_scenarios.py
```

Тесты:

```bash
pytest -q
```

Результаты сохраняются в `experiments/results/`.

## Ограничения прототипа

- Оптимизация аукциона реализована жадной аппроксимацией глобального распределения с учётом CPU/MEM ограничений узлов.
- Сценарии ориентированы на экспериментальную валидацию метода, а не на промышленный runtime.

## 📝 Лицензия

MIT License — свободное использование и модификация
