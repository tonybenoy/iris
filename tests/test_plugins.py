"""The provider layer has to be genuinely extensible, not an if/elif chain
wearing the word "plugin". These tests add a provider the way a user would.
"""
import pytest

from pa.config import Config
from pa.providers import registry
from pa.providers.base import Annotation, CaptionProvider


def test_a_third_party_captioner_needs_no_project_changes():
    @registry.register("caption", "my_captioner")
    class Mine(CaptionProvider):
        def __init__(self, cfg):
            self.cfg = cfg

        @property
        def model_version(self):
            return "mine@1"

        def annotate(self, image_bytes):
            return Annotation(caption="from my plugin")

    cfg = Config()
    cfg.caption.provider = "my_captioner"
    provider = registry.get_captioner(cfg)
    assert provider.annotate(b"").caption == "from my plugin"
    assert "my_captioner" in registry.available("caption")


def test_a_plugin_can_override_a_builtin():
    registry._resolve("caption", "lmstudio")          # ensure the built-in is loaded
    original = registry._REGISTRY["caption"]["lmstudio"]

    @registry.register("caption", "lmstudio")
    class Replacement(CaptionProvider):
        def __init__(self, cfg):
            pass

        @property
        def model_version(self):
            return "replaced"

        def annotate(self, image_bytes):
            return Annotation(caption="replaced")

    cfg = Config()
    cfg.caption.provider = "lmstudio"
    assert registry.get_captioner(cfg).model_version == "replaced"
    # Restore the real one. Popping instead would lose it for good: the module
    # is already in sys.modules, so re-importing will not re-run @register.
    registry._REGISTRY["caption"]["lmstudio"] = original
    assert registry.get_captioner(cfg).model_version != "replaced"


def test_wrong_base_class_is_refused():
    """A provider that does not implement the interface must fail at
    registration, not with an AttributeError deep inside the indexer."""
    with pytest.raises(TypeError, match="must subclass"):
        @registry.register("caption", "bogus")
        class NotAProvider:
            pass


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown provider kind"):
        registry.register("telepathy", "x")


def test_unknown_name_lists_what_is_available():
    cfg = Config()
    cfg.caption.provider = "does_not_exist"
    with pytest.raises(ValueError, match="Available:"):
        registry.get_captioner(cfg)


def test_plugin_loaded_from_a_loose_file(tmp_path):
    """The no-packaging path: point config at a .py file and it is imported."""
    plugin = tmp_path / "file_captioner.py"
    plugin.write_text(
        "from pa.providers.base import Annotation, CaptionProvider\n"
        "from pa.providers.registry import register\n"
        "\n"
        "@register('caption', 'from_file')\n"
        "class FileCaptioner(CaptionProvider):\n"
        "    def __init__(self, cfg): self.cfg = cfg\n"
        "    @property\n"
        "    def model_version(self): return 'file@1'\n"
        "    def annotate(self, image_bytes): return Annotation(caption='hello from a file')\n"
    )
    cfg = Config()
    cfg.caption.provider = "from_file"
    cfg.caption.plugins = [str(plugin)]
    assert registry.get_captioner(cfg).annotate(b"").caption == "hello from a file"


def test_builtins_are_imported_lazily():
    """Asking for a caption provider must not drag in torch."""
    import sys
    before = "torch" in sys.modules
    registry._resolve("caption", "lmstudio")
    assert ("torch" in sys.modules) == before, \
        "resolving a captioner should not import the ML stack"
