"""Native preview is required in both source collection and environment checks."""
import pytest
import bundle_paths
import runtime_environment


def test_native_presenter_is_required_and_its_interface_is_checked():
    assert 'gpu_native_presenter.dll' in bundle_paths.NATIVE_BINARIES
    assert runtime_environment.BRIDGES['gpu_native_presenter.dll'] == ('gpu_native_presenter_open', None)


def test_collecting_a_bundle_refuses_missing_presenter(tmp_path):
    native = tmp_path / 'native'
    native.mkdir()
    for name in bundle_paths.NATIVE_BINARIES:
        if name != 'gpu_native_presenter.dll':
            (native / name).write_bytes(b'fixture')
    tools = tmp_path / 'tools'
    tools.mkdir()
    for name in bundle_paths.TOOL_BINARIES:
        (tools / name).write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError, match='gpu_native_presenter.dll'):
        bundle_paths.resolve_sources(str(tmp_path), str(tools))
