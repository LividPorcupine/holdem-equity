import subprocess
import sys

import pytest

from holdem_equity.cli import main


def test_cli_monte_carlo(capsys):
    main(["AhKh", "QsQd", "--samples", "2000", "--seed", "1"])
    out = capsys.readouterr().out
    assert "AhKh vs QsQd" in out and "Monte Carlo, 2,000 samples, seed 1" in out
    assert out.count("%") >= 4


def test_cli_exact_with_board(capsys):
    main(["AhKh", "QsQd", "--board", "QhJhTh2c3d", "--exact"])
    out = capsys.readouterr().out
    assert "exact, 1 showdowns" in out and "100.00%" in out


def test_cli_random_villain(capsys):
    main(["AhAs", "random", "--samples", "2000", "--seed", "1"])
    assert "AhAs vs random" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["AhKh", "AhQd"], ["AhKh", "QsQd", "--board", "Td9h"], ["Ax", "QsQd"]])
def test_cli_rejects_bad_input(argv):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


def test_python_dash_m_entry_point():
    result = subprocess.run([sys.executable, "-m", "holdem_equity", "AhKh", "QsQd", "--samples", "1000", "--seed", "0"],
                            capture_output=True, text=True, check=True)
    assert "AhKh vs QsQd" in result.stdout
