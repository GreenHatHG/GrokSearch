import pytest
import httpx


class TestEmptyResultRetryIntegration:
    """测试空结果重试的集成测试"""

    @pytest.mark.asyncio
    async def test_default_config_retries_on_empty(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：默认配置（不设置环境变量），返回空结果
        预期：应该重试（默认启用）
        """
        responses = ["", "default config works"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 2, f"默认配置应该启用重试，实际调用了 {call_counter['count']} 次"
        assert "default config works" in result

    @pytest.mark.asyncio
    async def test_explicit_disabled_no_retry(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：显式禁用空结果重试，返回空结果
        预期：不应该重试
        """
        monkeypatch.setenv("GROK_RETRY_EMPTY_RESULTS", "false")

        responses = ["", "123"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 1, "禁用时不应该重试"
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_retry_when_content_present(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：第一次就返回有内容的结果
        预期：不应该重试
        """
        responses = ["good content"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 1, "有内容时不应该重试"
        assert "good content" in result

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_empty(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：一直返回空结果，重试次数用完
        预期：应该重试 N 次，最终返回空字符串
        """
        monkeypatch.setenv("GROK_RETRY_EMPTY_RESULTS", "true")
        monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "2")

        responses = ["", "", ""]

        with mock_http_client(responses):
            result = await provider.search("test query")

        # 验证：应该重试了 3 次（初始 1 次 + 重试 2 次）
        assert call_counter['count'] == 3, f"应该重试 2 次（总共 3 次请求），实际 {call_counter['count']} 次"
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_method_also_retries(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：测试 fetch() 方法也支持空结果重试
        预期：fetch() 方法应该和 search() 一样支持重试
        """
        responses = [" ", "fetched content"]

        with mock_http_client(responses):
            result = await provider.fetch("http://example.com")

        assert call_counter['count'] == 2, "fetch() 方法应该重试一次"
        assert "fetched content" in result

    @pytest.mark.asyncio
    async def test_network_error_triggers_retry(self, provider, mock_http_client, call_counter):
        """
        场景：第一次网络错误，第二次成功
        预期：应该重试（验证现有的网络错误重试机制仍然工作）
        """
        responses = [httpx.TimeoutException("timeout"), "success after retry"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 2, f"网络错误应该触发重试，实际调用了 {call_counter['count']} 次"
        assert "success after retry" in result

    @pytest.mark.asyncio
    async def test_network_error_and_empty_result_share_retry_count(self, monkeypatch, provider, mock_http_client,
                                                                    call_counter):
        """
        场景：第一次网络错误，第二次空结果，第三次成功
        预期：两种重试机制共享同一个计数器
        """
        monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "3")

        responses = [httpx.NetworkError("network error"), "", "final success"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 3, f"应该总共调用 3 次，实际 {call_counter['count']} 次"
        assert "final success" in result

    @pytest.mark.asyncio
    async def test_http_403_not_retry_by_default(self, provider, mock_http_client, call_counter):
        """
        场景：第一次返回 403 的 HTTPStatusError（默认配置）
        预期：不应该重试，直接抛出异常
        """
        request = httpx.Request("POST", "http://test-api.com/chat/completions")
        response = httpx.Response(403, request=request)
        exc = httpx.HTTPStatusError("403 Forbidden", request=request, response=response)

        responses = [exc, "should not be returned"]

        with mock_http_client(responses):
            with pytest.raises(httpx.HTTPStatusError):
                await provider.search("test query")

        assert call_counter['count'] == 1, f"默认不应重试 403，实际调用了 {call_counter['count']} 次"

    @pytest.mark.asyncio
    async def test_http_403_retries_when_configured(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：第一次返回 403 的 HTTPStatusError，但配置将 403 加入可重试列表
        预期：应该重试一次并成功
        """
        monkeypatch.setenv("GROK_RETRY_EXTRA_STATUS_CODES", "403")
        monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "1")
        monkeypatch.setenv("GROK_RETRY_MULTIPLIER", "0")
        monkeypatch.setenv("GROK_RETRY_MAX_WAIT", "0")

        request = httpx.Request("POST", "http://test-api.com/chat/completions")
        response = httpx.Response(403, request=request)
        exc = httpx.HTTPStatusError("403 Forbidden", request=request, response=response)

        responses = [exc, "success after 403 retry"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 2, f"开启后应重试 1 次（总共 2 次请求），实际 {call_counter['count']} 次"
        assert "success after 403 retry" in result

    @pytest.mark.asyncio
    async def test_http_multiple_status_codes_retries_when_configured(self, monkeypatch, provider, mock_http_client, call_counter):
        """
        场景：配置多个可重试状态码（逗号分隔），第一次返回其中一个（409）
        预期：应该重试一次并成功
        """
        monkeypatch.setenv("GROK_RETRY_EXTRA_STATUS_CODES", "403, 409")
        monkeypatch.setenv("GROK_RETRY_MAX_ATTEMPTS", "1")
        monkeypatch.setenv("GROK_RETRY_MULTIPLIER", "0")
        monkeypatch.setenv("GROK_RETRY_MAX_WAIT", "0")

        request = httpx.Request("POST", "http://test-api.com/chat/completions")
        response = httpx.Response(409, request=request)
        exc = httpx.HTTPStatusError("409 Conflict", request=request, response=response)

        responses = [exc, "success after 409 retry"]

        with mock_http_client(responses):
            result = await provider.search("test query")

        assert call_counter['count'] == 2, f"开启后应重试 1 次（总共 2 次请求），实际 {call_counter['count']} 次"
        assert "success after 409 retry" in result
