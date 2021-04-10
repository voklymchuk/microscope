from microscope.data.generic_processing import main


def test_main(runner):
    with open("hello.txt", "w") as f:
        f.write("Hello World!")
    result = runner.invoke(main, ["hello.txt", "output_path"])
    assert result.exit_code == 0
