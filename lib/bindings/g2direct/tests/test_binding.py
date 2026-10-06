import importlib.util
import sys
import unittest

from g2direct import HiCacheG2Direct


class StandaloneBindingTest(unittest.TestCase):
    def test_nixl_is_loaded_before_native_extension(self):
        modules = list(sys.modules)
        self.assertLess(modules.index("nixl"), modules.index("g2direct._core"))

    def test_import_without_kvbm_or_dynamo_python_runtime(self):
        self.assertIsNone(importlib.util.find_spec("kvbm"))
        self.assertIsNone(importlib.util.find_spec("dynamo"))
        for method in (
            "register_host_region",
            "publish",
            "unpublish",
            "lookup_prefix",
            "lookup_mask",
            "fetch",
            "stats",
        ):
            self.assertTrue(callable(getattr(HiCacheG2Direct, method)))

    def test_invalid_configuration_fails_before_discovery(self):
        for arguments, message in (
            ({"tp_world_size": 0}, "tp_world_size"),
            ({"tp_rank": 1}, "tp_rank"),
            ({"page_size": 0}, "page_size"),
            ({"max_transfer_mib": 0}, "max_transfer_mib"),
        ):
            with self.subTest(arguments=arguments):
                config = {"tp_rank": 0, "tp_world_size": 1, "page_size": 64}
                config.update(arguments)
                with self.assertRaisesRegex(ValueError, message):
                    HiCacheG2Direct(**config)


if __name__ == "__main__":
    unittest.main()
