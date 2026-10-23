"""File này cho phép chạy `python -m skq_exp` khi chưa cài console command.

Nó chuyển toàn bộ tham số sang CLI chính, vì vậy local, Kaggle và package đã cài
đều dùng cùng một đường chạy.
"""

from .cli import main

raise SystemExit(main())

