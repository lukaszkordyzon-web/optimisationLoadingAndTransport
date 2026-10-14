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
PYTHONPATH=. python tools/walidacja/warianty.py    # porównanie wariantów: 1 zmiana vs 20 zmian, 5 pytań × 40 powtórzeń
PYTHONPATH=. python tools/walidacja/kalibracja.py   # kalibracja z tabeli 20 zmian vs z jednej zmiany: prognoza 40 kolejnych
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
| swoja koparka (przydział sztygara), do której wracają po wyładunku | gdzie jest kolejka pod koparkami |
| nie zmieniają koparki sami – tylko na polecenie sztygara | ile wozideł jest w drodze i kiedy koparka będzie wolna |
| awaria koparki, koniec usypu (radio) – sztygar kieruje do innej | prognozy, zalecane prędkości, kiedy parkować |
| usyp się skończył, przerwa, strzelanie, koniec zmiany | ile ton jest w drodze (cel zmiany – dopiero gdy waga go pokaże) |
| przy kilku kruszarkach: najbliższa, chyba że widać wyraźnie dłuższą kolejkę | |

Zalecenia, parkowanie, pomijanie łyżki i odstawianie koparek działają tylko w trybie systemu (`dynamic`) i RL – nawet
jeśli w pliku kopalni zapisano nauczoną politykę (testy w `tests/test_walidacja.py`).

## Polityki na tych samych 30 zmianach

„Jak dziś” = przydział sztygara (kierowca nie zmienia koparki sam). Koszt na tonę liczony jako **wszystkie koszty /
wszystkie tony** z 30 zmian – średnia z zł/t pojedynczych zmian jest fałszywa, bo jedna zmiana z wielogodzinną
awarią kruszarki (250 t, ponad 100 zł/t) przeważa resztę.

| sytuacja | jak dziś | system (heurystyka) | RL | lepiej na | czego RL się nauczył |
|---|---|---|---|---|---|
| A – kruszarka gardłem (6 wozideł) | 19,54 | −0,8 % | **−6,6 %** | 30/30 | parkowanie: paliwo/t −38 %, kolejka pod koszem 26 h → 7 h |
| B – transport gardłem | 19,09 | −4,9 % | **−4,9 %** | 29/30 | wszystkie wozidła na bliższy poziom |
| C – zrównoważona | 15,86 | −4,2 % | **−4,9 %** | 29/30 | jw. – poziom 2 porzucony |
| D – jak C + skład nadawy 50/50 | 16,30 | **+5,5 %** (łamie skład: 90 % z B1) | **−0,8 %** | 28/30 | trzyma regułę (51/49) |

Wniosek: RL robi to, za co go nagradzamy. Bez reguł (skład nadawy, przydział 🔒, cel zmiany) „optymalizuje” rzeczy,
których kopalnia nie może zrobić (porzucić poziom). Z regułami trzyma się ich; gdy kruszarka jest wąskim gardłem,
realny zysk daje parkowanie wozideł, sam przydział przy dwóch koparkach i dobrym przydziale sztygara – niewiele.

## Kolejki pod koparkami i na mijankach (`kolejki.py`)

Rampy jednopasowe z mijanką w połowie, kruszarka 800 t/h (nie jest gardłem), koparki z łyżką 3 m³, 6 wozideł, skład
nadawy 50/50. Zatrzymanie na mijance kosztuje też ruszenie (rozpędzanie). Te same 30 zmian:

| | jak dziś (sztygar) | system (heurystyka) | RL – przydział | RL + zalecenia prędkości |
|---|---|---|---|---|
| tony na zmianę | 4 399 | 4 442 | 4 400 | **4 476 (+1,7 %)** |
| zł/t (koszty / tony) | 15,22 | −1,0 % | −1,3 % | **−2,1 %** (27/30 zmian) |
| wozidła w kolejce pod koparkami [h] | 4,9 | 4,3 | 4,4 | **3,6** |
| czekanie na mijankach [h] | 3,2 | 3,3 | 3,3 | 3,4 |
| łyżka nabrana przed podstawieniem | 7 | 8 | 8 | **32** |

Przy stałym przydziale sztygara kolejki pod koparkami są umiarkowane; system i RL skracają je o 10–25 %, najwięcej
daje zapowiedź dla operatora (łyżka gotowa, gdy wozidło podjeżdża). Losowe zmiany koparki przez kierowców (założenie
z pierwszej wersji) robiły kolejki 7,4 h – w rzeczywistości kierowcy jeżdżą tam, gdzie każe sztygar.
RL zna ruch na rampach (z planów jazdy wszystkich wozideł – w kopalni z GPS) i bierze go pod uwagę przy wyborze
koparki. Zalecenie „zjedź wolniej, miniesz się bez zatrzymania” sprawdzone i **odrzucone**: przy stałym strumieniu
pełnych wozideł pod górę liczbę spotkań wyznacza ruch i liczba mijanek.

Infrastruktura (te same 30 zmian, koszty / tony):

| rampa | jak dziś: t / zł/t | system: t / zł/t |
|---|---|---|
| 1 pas, 1 mijanka | 4 399 / 15,22 | 4 442 / 15,07 |
| 1 pas, 2 mijanki | 4 470 / 15,09 | 4 531 / 14,91 |
| 1 pas, 3 mijanki | 4 508 / 15,01 | 4 586 / 14,81 |
| 2 pasy | 4 610 / 14,81 | 4 639 / 14,71 |

Druga mijanka: +1,6 % ton; drugi pas: +4,8 %. Przy jednym pasie więcej daje infrastruktura niż sam przydział.

Zasady na odcinkach jednopasowych – takie same dla wozideł i aut osobowych (sztygar, ekipa strzałowa, cysterna):
jadący pod górę ma pierwszeństwo; kto już minął mijankę i jest na odcinku, jedzie pierwszy (jadący pod górę czeka na
mijance); nikt nie wjeżdża na odcinek, jeśli jadący z naprzeciwka dojedzie do mijanki, zanim zdążyłby przejechać.
Zatrzymanie kosztuje ruszenie. Test: nigdy dwa pojazdy w przeciwnych kierunkach na jednym odcinku.

## Porównanie wariantów na wielu zmianach (`warianty.py`)

Krok „Wariant – co jeśli” porównuje teraz wariant ze stanem obecnym na 10/20/30 tych samych zmianach i ocenia
wynik testem znaków (wariant „lepszy”, jeśli wygrywa na co najmniej 15 z 20 zmian), plus tabela „ile wozideł”.
Sprawdzenie: 5 pytań, każde zadane 40 razy na innych zmianach; prawda z 200 zmian.

| pytanie (prawda z 200 zmian) | jedna zmiana: dobrze / źle / nie wiadomo | 20 zmian: dobrze / źle / nie wiadomo |
|---|---|---|
| 4 zamiast 3 wozideł – taniej? (nie, +4,9 %) | 39 / 1 / 0 | **40 / 0 / 0** |
| 4 zamiast 3 wozideł – więcej ton? (tak, +1,4 %) | 39 / 0 / 1 | **40 / 0 / 0** |
| druga mijanka – więcej ton? (tak, +1,8 %) | 37 / 0 / 3 | **40 / 0 / 0** |
| łyżka 5 m³ zamiast 4,5 – więcej ton? (+0,1 % – praktycznie nic) | 4 / 0 / 36 | 0 / 0 / **40** („bez pewnej różnicy”) |
| wozidła 40 t zamiast 36 t – taniej? (tak, −0,6 %) | 38 / **2** / 0 | **39 / 0 / 1** |

Wielkość efektu (P10–P90 oszacowania z 40 powtórzeń):

| pytanie | jedna zmiana | 20 zmian |
|---|---|---|
| druga mijanka – ile więcej ton | +0,7 … +3,1 % | **+1,7 … +2,2 %** |
| wozidła 40 t – ile taniej | −0,9 … 0,0 % | **−0,8 … −0,5 %** |

Wnioski: porównanie na tych samych zmianach (ta sama pogoda, awarie, kierowcy) już przy jednej zmianie zwykle
wskazuje dobry kierunek, ale myli się w kilku procentach przypadków i podaje wielkość efektu z rozrzutem kilku
punktów procentowych. 20 zmian: w 200 próbach ani jednej złej odpowiedzi, wielkość efektu z dokładnością ~±0,3 pp,
a efekt pomijalny (łyżka +0,1 %) uczciwie nazwany „bez pewnej różnicy” zamiast „lepiej”.

## Kalibracja: tabela zmian vs jedna zmiana (`kalibracja.py`)

„Ukryta prawda” = kopalnia testowa z parametrami, których kalibracja nie zna (przestoje na kurs, zmienność tempa
między zmianami, wolniejsza jazda). Z niej 20 „zmierzonych” zmian (tony, kursy, paliwo), potem 40 przyszłych.
Model bazowy kalibrowany (a) na tabeli 20 zmian – średnia z awariami + typowy rozrzut, (b) na pojedynczych
zmianach (stary sposób) – i prognozuje przyszłe 40 zmian.

| ukryta prawda | przyszłe 40 zmian | tabela 20 zmian | jedna zmiana (5 prób) |
|---|---|---|---|
| przestoje 6 min, zmienność 12 % | 2 478 t | −1,0 % | −13,6 … +2,8 % |
| przestoje 3 min, bez zmienności | 2 642 t | −1,4 % | −10,8 … −3,6 % |
| przestoje 4 min, zmienność 20 % | 2 580 t | −0,7 % | −8,7 … +1,0 % |
| jazda wolniej o 15 %, przestoje 8 min, zmienność 10 % | 2 141 t | −0,1 % | −12,3 … +1,6 % |

Średni błąd prognozy: **tabela 0,8 % (maks. 1,4 %)**, jedna zmiana 5,2 % (maks. 13,6 %) – zależy od tego, którą
zmianę ktoś wpisał. Zakres P10–P90 z tabeli trafia w zakres przyszłych zmian (zmienność 20 % – nieco węższy).
Rozrzut liczony odpornie (rozstęp ćwiartkowy / 1,349): odchylenie standardowe 20 zmian potrafi się podwoić od
jednej zmiany z długą awarią kruszarki.

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
6. Kierowcy w „jak dziś” zmieniali koparkę sami w 25 % kursów – nierealne; teraz jeżdżą tylko tam, gdzie przypisał
   ich sztygar (zmiana na polecenie: awaria, koniec usypu). Da się to zmienić w kroku „Plan zmiany”.
7. System i RL nie wiedziały o awarii koparki (przewidywały, że jest wolna) – wysyłały tam wozidła; teraz wiedzą
   (co najmniej tyle, co ludzie z radia), a planowane postoje (przerwy, strzelanie, tankowanie) są w prognozie.
8. Koszt na tonę z wielu zmian (uczenie RL, sprawdzenie przed zapisem polityki, raporty) – wszystkie koszty / wszystkie
   tony zamiast średniej z zł/t zmian.
