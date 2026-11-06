# Карточки методов: source настройка и свежий тест

Все source choices сделаны до нового target доступа. Значения и exact selectors находятся в source_selection.csv и hypothesis_tests.csv; полный provenance — evidence_manifest.json.

## anchor / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=anchor в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=2.76657e-05; CI=[1.18212e-05; 4.35101e-05]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=anchor](hypothesis_tests.csv).
mechanism: mean=-4.92083e-05; CI=[-0.000329197; 0.000230781]; Holm p=1; status=complete. [Точная строка: hypothesis=mechanism, task=classification, method=anchor](hypothesis_tests.csv).

## crf / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=crf в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.0350341; CI=[0.0317975; 0.0382708]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=crf](hypothesis_tests.csv).
mechanism: mean=-0.00108649; CI=[-0.00154222; -0.000630757]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=mechanism, task=classification, method=crf](hypothesis_tests.csv).

## extra_trees / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=extra_trees в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.00300226; CI=[0.00233086; 0.00367365]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=extra_trees](hypothesis_tests.csv).

## hist_gbdt / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=hist_gbdt в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.000692688; CI=[-0.000229363; 0.00161474]; Holm p=1; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=hist_gbdt](hypothesis_tests.csv).

## irf / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=irf в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.0350306; CI=[0.0327961; 0.037265]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=irf](hypothesis_tests.csv).
mechanism: mean=-0.000430632; CI=[-0.000640925; -0.00022034]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=mechanism, task=classification, method=irf](hypothesis_tests.csv).

## meta / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=meta в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.00284956; CI=[0.00228028; 0.00341884]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=meta](hypothesis_tests.csv).
mechanism: mean=0.00237246; CI=[0.00198217; 0.00276274]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=mechanism, task=classification, method=meta](hypothesis_tests.csv).

## rf / classification

Контексты source HPO: 9. Все строки фильтруются task=classification, method=rf в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.00376569; CI=[0.00336191; 0.00416947]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=classification, method=rf](hypothesis_tests.csv).

## anchor / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=anchor в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0; CI=[0; 0]; Holm p=1; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=anchor](hypothesis_tests.csv).
mechanism: mean=-0.0041625; CI=[-0.0121078; 0.00378277]; Holm p=1; status=complete. [Точная строка: hypothesis=mechanism, task=regression, method=anchor](hypothesis_tests.csv).

## directional_era / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=directional_era в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.000232153; CI=[-0.00310619; 0.0035705]; Holm p=1; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=directional_era](hypothesis_tests.csv).
mechanism: mean=-0.00289297; CI=[-0.00491141; -0.000874534]; Holm p=0.875; status=complete. [Точная строка: hypothesis=mechanism, task=regression, method=directional_era](hypothesis_tests.csv).

## era / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=era в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.0306527; CI=[0.0252067; 0.0360986]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=era](hypothesis_tests.csv).
mechanism: mean=0.0148987; CI=[0.0109572; 0.0188402]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=mechanism, task=regression, method=era](hypothesis_tests.csv).

## extra_trees / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=extra_trees в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.00548895; CI=[0.00227085; 0.00870706]; Holm p=0.193359; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=extra_trees](hypothesis_tests.csv).

## hist_gbdt / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=hist_gbdt в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.0321844; CI=[0.0228493; 0.0415195]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=hist_gbdt](hypothesis_tests.csv).

## irf / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=irf в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.058144; CI=[0.049008; 0.0672801]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=irf](hypothesis_tests.csv).
mechanism: mean=7.51919e-05; CI=[-8.5505e-05; 0.000235889]; Holm p=1; status=complete. [Точная строка: hypothesis=mechanism, task=regression, method=irf](hypothesis_tests.csv).

## maxrm / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=maxrm в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.0679954; CI=[0.0599163; 0.0760745]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=maxrm](hypothesis_tests.csv).
mechanism: mean=-0.000338011; CI=[-0.00059319; -8.28312e-05]; Holm p=0.875; status=complete. [Точная строка: hypothesis=mechanism, task=regression, method=maxrm](hypothesis_tests.csv).

## rf / regression

Контексты source HPO: 8. Все строки фильтруются task=regression, method=rf в [source selection](source_selection.csv). Параметры каждого контекста сохранены целиком; единственный универсальный лучший config не заявляется.
HPO: mean=0.00429709; CI=[0.00214493; 0.00644925]; Holm p=0.046875; status=complete. [Точная строка: hypothesis=HPO, task=regression, method=rf](hypothesis_tests.csv).
