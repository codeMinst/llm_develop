"""
기능 객체 또는 함수를 일관된 .run(x) 인터페이스로 감싸는 유틸리티입니다.
None이 들어오면 no-op 처리기로 대체됩니다.
"""

from types import SimpleNamespace
from typing import Any, Callable, Union

class Runner:
    @staticmethod
    def from_callable(fn: Callable, name: str = "anonymous") -> SimpleNamespace:
        return SimpleNamespace(
            run=fn,
            get_feature_name=lambda: name
        )

    @staticmethod
    def from_object(obj: Any, name: str = "anonymous") -> SimpleNamespace:
        if hasattr(obj, "run"):
            return obj
        raise TypeError(f"Object of type {type(obj)} does not have a .run() method.")

    @staticmethod
    def disabled(name: str = "noop") -> SimpleNamespace:
        return SimpleNamespace(
            run=lambda x: x,
            get_feature_name=lambda: name
        )

    @staticmethod
    def wrap(obj: Union[Callable, Any, None], name: str = "anonymous") -> SimpleNamespace:
        """
        주어진 obj를 .run(x) 인터페이스로 래핑하여 반환합니다.
        - None: 비활성화 처리기 (lambda x: x)
        - .run() 메서드를 가진 객체: 그대로 반환
        - 일반 함수: run() 인터페이스로 감쌈
        """
        if obj is None or obj is False:
            return Runner.disabled(name)
        if hasattr(obj, "run"):
            return Runner.from_object(obj, name)
        if callable(obj):
            return Runner.from_callable(obj, name)
        raise TypeError(f"지원하지 않는 feature 타입: {type(obj)}")
