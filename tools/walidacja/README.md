# Walidacja modeli na kopalni testowej

Jedna kopalnia, jeden schemat – zmieniane są tylko parametry, żeby zobaczyć, jak reagują symulacja zmiany,
polityka RL i harmonogram.

```
kruszarka C-1 ── 400 m prosto ── rampa 1 (600 m, 8 %) ── poziom 1: 250 m ── odstrzał B1 (koparka K-1)
                                                      └─ rampa 2 (500 m, 8 %) ── poziom 2: 250 m ── odstrzał B2 (koparka K-2)
```

Dobór „jak inżynier” (`kopalnia_testowa.human_sizing`): kruszarka 400 t/h (przy P80 ~450 mm efektywnie 352 t/h),
2 koparki 70 t z łyżką 4,5 m³, wozidła 36 t. Cykl: poziom 1 – 11,6 min, poziom 2 – 15,6 min, ok. 160 t/h na wozidło
→ **3 wozidła**. Symulacja daje to samo: 3 wozidła to najmniejsza flota, przy której kruszarka jest pełna.

```
PYTHONPATH=. python tools/walidacja/czulosc.py     # 1 400 zmian: 10 parametrów po kolei + siatka wozidła × kruszarka × odległość
PYTHONPATH=. python tools/walidacja/polityki.py    # 4 sytuacje × nauka RL + 4 polityki × 30 tych samych zmian
PYTHONPATH=. python tools/walidacja/kolejki.py     # rampy jednopasowe, kolejki pod koparkami: 4 polityki × 30 zmian
PYTHONPATH=. python tools/walidacja/harmonogram.py # miesiąc: flota, wyczerpanie usypów, strzały, szansa realizacji, cel od kruszarek
```

Razem ok. **6 000 symulowanych zmian** (z uczeniem RL i 40 wariantami miesiąca). Każda zmiana jest też sprawdzana
pod kątem fizyki: bilans masy, wykorzystanie ≤ 100 %, dodatni koszt.

## Co wyszło (średnie z 20 zmian na punkt)

| parametr | zakres | tony na zmianę | zł/t | ocena |
|---|---|---|---|---|
| wozidła | 1 → 8 | 1 276 → 2 714 (plateau od 3–4) | 23,0 → **16,9 (min. przy 3)** → 21,3 | ✅ nasycenie na kruszarce, potem tylko kolejki |
| kruszarka | 250 → 800 t/h | 1 739 → 3 505 (plateau od 600) | 21,7 → 14,7 | ✅ powyżej 600 t/h gardłem staje się transport |
| łyżka | 2,5 → 6,5 m³ | +2 % | −1 % | ✅ mały wpływ – załadunek nie jest gardłem |
| nachylenie ramp | 4 → 12 % | −2 % | +3 %, paliwo/t +25 % | ✅ |
| odległość | ×0,5 → ×3 | 2 731 → 1 640 | 16,3 → 23,9 | ✅ |
| ładowność wozidła | 28 → 55 t | +3 % | −4 %, paliwo/t −33 % | ✅ |
| rozdrobnienie | 0,1 → 0,9 | −31 % | +18 %, kWh/t +45 % | ⚠️ wpływ duży – założenie „−40 % wydajności kruszarki przy złym urobku” wymaga kalibracji |
| deszcz | 0 → 100 % zmian | −0,1 % | +0,5 % | ✅ (kruszarka gardłem maskuje) |
| awarie | wył./wł. | −1,3 % | +0,8 % | ✅ ale to one robią prawie cały rozrzut (sd 14 t → 97 t) |
| przestoje z kalibracji | 0 → 10 min/kurs | −23 % | +17 % | ✅ |

Najtańsza flota rośnie z wydajnością kruszarki i odległością (2 → 7 wozideł) – tak, jak dobrałby człowiek.

## „Jak dziś” = decyzje ludzi bez informacji

Symulacja stanu obecnego (krok „Jak było dziś”, kalibracja, wariant, baza harmonogramu i szansy realizacji) liczy
zawsze w trybie `natural`. Kierowca i operator wiedzą tylko to, co widzą i słyszą przez radio:

| wiedzą | nie wiedzą |
|---|---|
| swoja koparka (przydział sztygara), z którą wracają po wyładunku | gdzie jest kolejka pod koparkami |
| czasem (25 %) sami jadą do innej – losowo | ile wozideł jest w drodze i kiedy koparka będzie wolna |
| awaria koparki (radio) – jadą do innej, losowo | prognozy, zalecane prędkości, kiedy parkować |
| usyp się skończył, przerwa, strzelanie, koniec zmiany | ile ton jest w drodze (cel zmiany – dopiero gdy waga go pokaże) |
| przy kilku kruszarkach: najbliższa, chyba że widać wyraźnie dłuższą kolejkę | |

Zalecenia, parkowanie, pomijanie łyżki i odstawianie koparek działają tylko w trybie systemu (`dynamic`) i RL – nawet
jeśli w pliku kopalni zapisano nauczoną politykę (testy w `tests/test_walidacja.py`).

## Polityki na tych samych 30 zmianach (zł/t; różnica do „jak dziś”)

| sytuacja | jak dziś | system (heurystyka) | RL | czego RL się nauczył |
|---|---|---|---|---|
| A – kruszarka gardłem (6 wozideł) | 19,57 | −0,12 | **−1,08** (29/30 zmian) | parkowanie: paliwo/t −38 %, kolejka 26 h → 7 h |
| B – transport gardłem | 19,43 | −1,08 | **−1,15** (29/30) | wszystkie wozidła na bliższy poziom |
| C – zrównoważona | 16,15 | −0,87 | **−0,91** (29/30) | jw. – poziom 2 porzucony |
| D – jak C + skład nadawy 50/50 | 16,46 | **+0,85** (łamie skład: 91 % z B1) | **−0,05** (27/30; skład 51/49) | trzyma regułę; zysk z przydziału mały |

Wniosek: RL robi to, za co go nagradzamy. Bez reguł (skład nadawy, przydział 🔒, cel zmiany) „optymalizuje” rzeczy,
których kopalnia nie może zrobić (porzucić poziom). Z regułami trzyma się ich; gdy kruszarka jest wąskim gardłem,
realny zysk daje parkowanie wozideł, sam przydział przy dwóch koparkach – niewiele (w aplikacji taka polityka nie
jest zapisywana, bo zysk < 0,5 %).

## Kolejki pod koparkami i na mijankach (`kolejki.py`)

Rampy jednopasowe z mijanką w połowie, kruszarka 800 t/h (nie jest gardłem), koparki z łyżką 3 m³, 6 wozideł, skład
nadawy 50/50. Średnio z tych samych 30 zmian:

| | jak dziś | system (heurystyka) | RL – przydział | RL + zalecenia prędkości |
|---|---|---|---|---|
| tony na zmianę | 4 138 | 4 451 | 4 442 | 4 384 |
| zł/t | 19,33 | −0,56 | −0,58 | **−1,07** (28/30 zmian) |
| wozidła w kolejce pod koparkami [h] | **7,4** | 4,2 | 4,0 | **3,7** |
| czekanie na mijankach [h] / zatrzymań | 2,6 / 140 | 3,0 / 163 | 3,1 / 174 | 2,9 / 177 |
| łyżka nabrana przed podstawieniem | 8 | 8 | 9 | **36** |

Kolejki pod koparkami (2–3 wozidła pod jedną, gdy druga czeka) system i RL zmniejszają o połowę. Czekanie na
mijankach **nie maleje** – więcej kursów to więcej spotkań na rampie, a żadna polityka nie bierze pod uwagę ruchu
z naprzeciwka (to następny krok: cecha „kto jest na rampie” i zalecenie, kiedy ruszyć).

## Harmonogram

* miesiąc 40 zmian: najtańsza flota 3 wozidła (jak na pojedynczej zmianie);
* dwa usypy po 15 000 t bez strzałów → produkcja staje na 12. zmianie (30 000 t / ~2 600 t) ✅;
  ze strzałami co 2 dni – nie staje ✅;
* szansa realizacji: P10 102 800 t, P50 104 600 t, P90 106 500 t; średnio ~2 % mniej niż „zmiana × 40” – awarie
  przechodzące między zmianami ✅;
* cel od kruszarek (LP): przepływy i liczba wozideł sensowne; **koszt pokazywał tylko część zmienną (9,8 zł/t)** –
  poprawione: pokazujemy koszt pełny (25 zł/t produktu ≈ 21 zł/t nadawy), liczony jak w symulacji.

## Znalezione i poprawione błędy

1. Grupa identycznych wozideł (jeden wiersz „4 × …”) – każde miało „swoją” koparkę K-1, więc „jak dziś” prawie nie
   urabiało drugiego poziomu (88 % z B1). Teraz koparki są rozdzielane po kolei.
2. Ostatni kurs na koniec zmiany: urobek był już w koszu i kruszony, ale nie liczył się jako wywieziony (+2 t „z
   niczego”). Teraz kurs liczy się w chwili wsypania do kosza.
3. Koszt celu produkcyjnego (LP) – tylko część zmienna; teraz koszt pełny.
4. „Jak dziś” mogło korzystać z zaleceń prędkości, jeśli w kopalni zapisano nauczoną politykę (tryb zaawansowany) –
   teraz wiedza systemu działa tylko w trybie systemu i RL.
5. Przy awarii „swojej” koparki kierowcy czekali pod nią godzinami – teraz (radio) jadą do innej. Baza „jak dziś”
   jest przez to lepsza o ~2 %, a przewaga RL mniejsza – i uczciwa.
