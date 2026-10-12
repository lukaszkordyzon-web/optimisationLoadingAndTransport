# Kruszyk – optymalizacja załadunku, transportu i kruszenia

System do zarządzania łańcuchem **odstrzał → koparka → wozidła → rampa → kruszarka** na kopalni
kruszywa. Docelowo: aplikacja na telefony służbowe (pozycja wozideł, ETA pod koparkę, automatyczne
przydzielanie wozideł – bez dyspozytora) + cyfrowy bliźniak + modele ML.

**Etap obecny: dane syntetyczne.** Kopalnię opisuje się w pliku YAML (dowolna liczba koparek,
wozideł, odstrzałów, tras z rampami, parametry kruszarki), a symulator generuje dane w takim
kształcie, w jakim później przyjdą z telefonów (przejazdy odcinków między geofence'ami, stany wozideł,
załadunki, poziom kosza).

## Szybki start

```bash
pip install -r requirements.txt

# Panel w przeglądarce (mapa, scenariusze, wyniki, podgląd na żywo, porównania)
streamlit run app.py

# 1. Kalkulator analityczny: czasy cyklu, match factor, wąskie gardło, ile wozideł
python -m kruszyk calc scenarios/kruszyk_bazowy.yaml

# 2. Symulacja jednej zmiany (+ zapis danych syntetycznych do CSV)
python -m kruszyk simulate scenarios/kruszyk_bazowy.yaml --out out/zmiana1
python -m kruszyk simulate scenarios/kruszyk_bazowy.yaml --trucks 6 --dispatch fixed

# 3. Dobór liczby wozideł symulacją (koszt na tonę, kolejki, wykorzystanie)
python -m kruszyk sweep scenarios/kruszyk_bazowy.yaml --min 3 --max 12 --target-tph 900

# 4. Dane z wielu zmian + model ML czasów przejazdu i ETA pod koparkę
python -m kruszyk train scenarios/kruszyk_bazowy.yaml --shifts 40 --out data/segments.csv

python -m pytest
```

## Panel (`streamlit run app.py`)

| Zakładka | Co robi |
|---|---|
| 🗺️ Mapa kopalni | Kalibracja (link z Google Maps lub współrzędne), rysowanie na zdjęciu satelitarnym: drogi i rampy (nachylenie, limit), usypy (materiał, tonaż, rozdrobnienie, gęstość), koparki, kruszarka. Przycisk **„Zbuduj trasy”** wyznacza najkrótszą drogę każda koparka → kruszarka i wstawia ją do scenariusza |
| ⚙️ Scenariusz | Edycja wszystkich parametrów w tabelach + szybki kalkulator match factor; zapis do YAML |
| 📊 Wyniki zmiany | KPI, koparki, oś czasu wozideł (gdzie czekają), kosz kruszarki i hałda, struktura kosztu, eksport CSV |
| 📍 Podgląd na żywo | Suwak czasu: wozidła na mapie i na profilu trasy, **ekran operatora koparki** (kto pod łyżką, kto czeka, kto nadjeżdża i za ile), **ekrany kierowców** (polecenia), animacja zmiany |
| ⚖️ Porównanie wariantów | Liczba wozideł × warianty (przydział stały/automatyczny, hałda, doradztwo prędkości) z rekomendacją; porównanie zapisanych scenariuszy |
| 💥 Odstrzał ↔ kruszarka | Mine-to-mill: koszt całkowity na tonę w funkcji rozdrobnienia (droższy odstrzał vs szybszy załadunek i kruszenie) |
| 🤖 Model ETA | Trening modelu czasów przejazdu na danych z wielu zmian, kalkulator ETA |

Podkład mapy: zdjęcia satelitarne Esri World Imagery + OpenStreetMap (bez klucza API).
Nachylenia dróg podaje się ręcznie – w kierunku do kruszarki, „+” = pod górę z ładunkiem
(zdjęcie satelitarne nie zawiera wysokości). Końce dróg, które się łączą, muszą być ≤ 30 m od siebie.

## Co jest w środku

| Moduł | Rola |
|---|---|
| `kruszyk/config.py` | Scenariusz: odstrzały, koparki, floty wozideł, trasy (odcinki z długością i nachyleniem), kruszarka, zmienność, koszty |
| `kruszyk/physics.py` | Czasy oczekiwane: prędkość wozidła z mocy i oporów (nachylenie + opór toczenia), hamowanie na zjeździe, liczba łyżek, czas załadunku, wydajność kruszarki zależna od rozdrobnienia |
| `kruszyk/analytics.py` | Match factor, potrzebna liczba wozideł, wąskie gardło (koparki / wozidła / kruszarka) |
| `kruszyk/sim.py` | Symulacja zdarzeń dyskretnych (SimPy) – cyfrowy bliźniak; kolejki pod koparką i kruszarką, kosz kruszarki, hałda buforowa z ładowarką, doradztwo prędkości, awarie, deszcz, styl kierowcy, paliwo, energia, koszty |
| `kruszyk/optimize.py` | Przegląd liczby wozideł z replikacjami, rekomendacja (min. PLN/t, opcjonalnie z celem t/h) |
| `kruszyk/site.py` | Mapa: geometria (odległości, pola), parsowanie linków Google Maps, graf dróg, najkrótsza trasa, mapa → scenariusz |
| `kruszyk/views.py` | Stan wozideł w danej chwili, pozycja na trasie/mapie, ETA, ekran operatora i polecenia kierowców |
| `kruszyk/ml.py` | Generowanie zbioru danych, model czasu przejazdu odcinka (gradient boosting), ETA wozidła pod koparkę |

### Dyspozycja bez dyspozytora

`dispatch: dynamic` – system sam decyduje, do której koparki jedzie wozidło po wyładunku:
wybiera tę, pod którą wozidło **najwcześniej zostanie załadowane**, uwzględniając czas dojazdu
i wozidła już wysłane do każdej koparki. To ta sama logika, która trafi do aplikacji kierowcy
(„jedź do K-2”). `dispatch: fixed` – wozidło przypisane na stałe do koparki (stan obecny, do porównań).

### Kluczowe wskaźniki (KPI)

- wywiezione / pokruszone t, wydajność t/h, liczba cykli
- wykorzystanie koparek i ich czas oczekiwania na wozidła, moment wybrania odstrzału
- kolejki wozideł pod koparką i pod kruszarką (h, min/cykl), awarie, postój bez urobku
- wykorzystanie kruszarki, czas „głodzenia” kruszarki (pusty kosz)
- koszt na tonę

### Dane syntetyczne (format zgodny z przyszłą aplikacją)

- `segments.csv` – każdy przejazd odcinka: wozidło, kurs, koparka, odcinek, pełny/pusty, długość,
  nachylenie, deszcz, godzina zmiany, czas rzeczywisty (`duration_s`) i fizyczny (`physics_s`)
- `states.csv` – oś czasu stanów wozidła: `jazda_pusty`, `kolejka_koparka`, `podstawienie`,
  `zaladunek`, `jazda_pelny`, `kolejka_kruszarka`, `wyladunek`, `awaria`, `postoj_brak_urobku`
- `loads.csv` – załadunki (t, koparka, odstrzał, czas załadunku i wyładunku)
- `hopper.csv` – poziom kosza kruszarki co minutę

## Przykładowy wniosek ze scenariusza bazowego

2 koparki mogłyby razem przerobić ~1370 t/h, ale kruszarka przy danym rozdrobnieniu bierze ~1000 t/h.
Od 6 wozideł wydajność już nie rośnie – każde kolejne wozidło dokłada tylko kolejkę pod koszem
i ok. 0,46 PLN/t kosztu. Wniosek: przy obecnych założeniach optymalizować trzeba kruszarkę
i rozdrobnienie z odstrzału, a nie dokładać wozideł.

## Plan dalszy

1. **Aplikacja mobilna (Android)** – GPS w tle, geofence'y stref (front, rampa, droga, kosz),
   automatyczne wykrywanie etapu cyklu, praca offline. Widok kierowcy: „jedź do K-x”, przyciski
   awaria/przerwa. Widok operatora koparki: nadjeżdżające wozidła z ETA.
2. **Backend** – przyjmowanie telemetrii, ten sam silnik dyspozycji co w symulatorze, panel KPI.
3. **ML na danych rzeczywistych** – ten sam `ml.py`, inne źródło danych; ETA, czas załadunku,
   wykrywanie anomalii (wozidło wolniejsze niż zwykle → możliwa usterka).
4. **Kruszenie** – sterowanie rytmem przyjazdów pod poziom kosza (stałe zasilanie kruszarki),
   sprzężenie rozdrobnienia z odstrzału z wydajnością koparki i kruszarki (mine-to-mill),
   zdjęcie urobku telefonem → ocena rozdrobnienia.

Uwaga prawna: śledzenie pozycji telefonów pracowników to monitoring (art. 22³ KP, RODO) –
cel, zakres (tylko zmiana i teren zakładu, pozycja przypisana do wozidła), informacja dla pracowników.
