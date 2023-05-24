class TestFailureListener:
    def pytest_runtest_makereport(self, item, call):
        if call.when == "call" and call.excinfo is not None:
            item.config.test_failed = True


def pytest_configure(config):
    config.test_failed = False
    config.pluginmanager.register(TestFailureListener(), "test_failure_listener")
