# Kruszyk – optymalizacja załadunku, transportu i kruszenia

System do zarządzania łańcuchem **odstrzał → koparka → wozidła → rampa → kruszarka** na kopalni
kruszywa. Docelowo: aplikacja na telefony służbowe (pozycja wozideł, ETA pod koparkę, automatyczne
przydzielanie wozideł – bez dyspozytora) + cyfrowy bliźniak + modele ML.

**Etap obecny: dane syntetyczne.** Kopalnię opisuje się w pliku YAML (dowolna liczba koparek,
wozideł, odstrzałów, tras z rampami, parametry kruszarki), a symulator generuje dane w takim
kształcie, w jakim później przyjdą z telefonów (przejazdy odcinków między geofence'ami, stany wozideł,
załadunki, poziom kosza).

## Kruszyk 2 – miesiąc z sygnałami RL i bez (nowa aplikacja)

```bash
streamlit run kruszyk2.py
```

Cztery kroki: **① Kopalnia** (mapa: drogi, kruszarka, koparki, obszary wydobycia; maszyny) → **② Harmonogram**
(Ty ustawiasz pracę: koparka – obszar i tony na dzień, wozidło – do której koparki albo „nie jeździ”; program numeruje
usypy obszaru – na start leży /1, każdy odstrzał to kolejny – dolicza wiercenie i strzał w porę i pokazuje konflikty) → **③ Symulacja** (ta sama zmiana obok
siebie: jak dziś i z sygnałami jazdy, z komunikatami dla kierowców) → **④ Wynik** (cały miesiąc). Oba miesiące idą
według tego samego, zatwierdzonego harmonogramu – różnią się tylko sposobem jazdy. Silnik: `kruszyk/` (wspólny
z wersją 1), planowanie: `kruszyk/workplan.py` i `kruszyk/schedule.py`, ekrany: `app2/`. Na Streamlit Cloud: *Main file path* = `kruszyk2.py`.
Wersja 1 (`app.py`) zostaje – opis w `ARCHIWUM.md`.

## Szybki start

```bash
pip install -r requirements-dev.txt   # runtime + pytest

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

## Wdrożenie na Streamlit Community Cloud

1. Wejdź na https://share.streamlit.io i zaloguj się kontem GitHub (`lukaszkordyzon-web`).
2. **Create app → Deploy a public app from GitHub** (albo z prywatnego repo – trzeba dać Streamlitowi dostęp).
3. Ustaw:
   - Repository: `lukaszkordyzon-web/optimisationLoadingAndTransport`
   - Branch: `claude/confident-wright-xg92g2` (albo `main` po scaleniu)
   - Main file path: `app.py`
   - Advanced settings → Python: **3.12** lub **3.13**
4. **Deploy** – po ~2–3 minutach aplikacja jest pod adresem `https://<nazwa>.streamlit.app`.

Każdy push na wybraną gałąź automatycznie aktualizuje aplikację. Zależności: `requirements.txt`
(przypięte wersje), konfiguracja: `.streamlit/config.toml`. Jeśli repo jest prywatne, w ustawieniach
aplikacji (Sharing) można ograniczyć dostęp do wskazanych adresów e-mail.

## Panel – tryb prosty (domyślny), 4 kroki

1. **① Kopalnia – stan obecny**: lokalizacja, rysowanie dróg i ramp (nachylenie, limit prędkości, drogi planowane),
   usypów (materiał, ilość, **krzywa uziarnienia**), koparek, kruszarki i **miejsca strzelania**; sieć dróg i poziomy;
   **zmiana i zdarzenia** (rozruch, przerwy, strzelanie, objazd sztygara, tankowanie, praktyka prędkości, mijanie);
   wozidła i kruszarka; „Zapisz kopalnię”.
2. **② Jak jest dziś** – symulacja zmiany przy obecnej praktyce, wskaźniki, „gdzie tracimy”, animacja na mapie.
3. **③ Wariant – co jeśli** – kopia stanu obecnego: zmieniasz wozidła, przypisania, drogi (otwierasz planowaną, zamykasz
   istniejącą), kruszarkę, uziarnienie, godziny zdarzeń – i widzisz wynik vs dziś.
4. **④ Optymalizacja** – uczenie ze wzmocnieniem przydziału wozideł + **zalecenia w czasie rzeczywistym**:
   prędkość dla każdego kierowcy (dojazd dokładnie, gdy koparka będzie wolna) i komunikaty dla operatora koparki
   („WZ-03 za 1:40 → nabierz łyżkę i czekaj”). Wynik vs dziś, krzywa uczenia, animacja z ekranami operatorów.

### Drogi, trasy dedykowane, miejsca postoju, plik scenariusza

- **Kierunek rysowania drogi**: przy rysowaniu wybierasz ⬇ w dół / ⬆ pod górę / ➡ płasko i nachylenie; na mapie
  strzałka pokazuje kierunek rysowania (zielona = wznosi się, pomarańczowa = opada). Rzędne liczone są z faktycznego
  przebiegu dróg – droga od kruszarki może też iść pod górę. Stare mapy są przeliczane automatycznie.
- **Trasy dedykowane** (tryb „Trasa dedykowana”): dla ekipy strzałowej, sztygara, cysterny albo przejazdu koparki.
  Bez trasy pojazd jedzie najkrótszą drogą po sieci. Tam, gdzie trasa biegnie po drodze wozideł, obowiązuje mijanie.
- **Miejsce postoju koparki** (znacznik 🅿️): tu koparka zjeżdża na strzelanie (opcjonalnie też na tankowanie) –
  trasą dedykowaną, najkrótszą drogą albo na wprost; bez znacznika cofa się o zadaną odległość swoją trasą.
  Czas przejazdu wydłuża postój; wozidła przestają zjeżdżać, zanim pierwsza koparka ruszy na postój.
- **📁 Mój scenariusz** (panel boczny): „Zapisz scenariusz do pliku” / „Wczytaj scenariusz z pliku” – cała kopalnia,
  zdarzenia, wariant i nauczona polityka w jednym pliku `.kruszyk.yaml`.

### Czego uczy się agent RL

| Obszar | Czego się uczy | Start nauki |
|---|---|---|
| Przydział wozideł | do której koparki po wyładunku (8 cech: kolejka, dojazd, odwóz, załadunek, urobek, wozidła w drodze, „własna” koparka, niedobór usypu względem udziału w nadawie) | przydział przypadkowy |
| Ile koparek pracuje | próg „min. wolnych wozideł na koparkę” – poniżej odstawia koparkę z najdłuższym cyklem (przegląd co 30 min) | wszystkie pracują |
| Ostatnia łyżka | przy jakim wypełnieniu i jakiej kolejce puścić wozidło bez ostatniej łyżki | zawsze do pełna |
| Zalecenia | zapas przyjazdu pod koparkę, maks. zwolnienie, zwalnianie przed kruszarką | reguły |

**Udział w nadawie** (przy usypie, opcjonalnie): docelowy procent nadawy z usypu. Liczony na bieżąco, odchyłka
kosztuje (`blend_penalty_per_t_pp`), agent steruje nim **tylko wozidłami nieprzypisanymi na sztywno**.
Odstawiona koparka kosztuje `excavator_idle_share` stawki (operator). Koparek obsługujących wozidła przypisane na
sztywno agent nie odstawia.

### Zdarzenia na zmianie

| Zdarzenie | Jak działa w symulacji |
|---|---|
| Rozruch / koniec | pierwszy kurs po rozruchu; ostatni kurs musi się skończyć przed „końcem pracy” |
| Przerwy | wozidła kończą kurs i stoją, koparki nie zaczynają załadunku |
| Strzelanie | od ewakuacji do otwarcia nikt nie wjeżdża do wyrobiska, koparki stoją; samochód ekipy jeździ po drogach |
| Objazd sztygara | samochód od kruszarki przez kolejne koparki i z powrotem |
| Cysterna | objeżdża koparki o podanych godzinach, każda stoi na czas tankowania |
| Tankowanie wozideł | według zużycia (jazda, postoje); poniżej rezerwy zjazd na stację |
| Mijanie | jadący w dół ustępuje jadącemu w górę: wozidło zjeżdżające traci czas, gdy auto wyjeżdża |
| Limity prędkości | obowiązujące na każdej drodze; zadany odsetek kierowców jeździ o X % szybciej – liczone przekroczenia |

## Panel – tryb zaawansowany (`streamlit run app.py`, przełącznik w panelu bocznym)

| Zakładka | Co robi |
|---|---|
| 🗺️ Mapa kopalni | Kalibracja (link z Google Maps lub współrzędne), rysowanie na zdjęciu satelitarnym: drogi i rampy (nachylenie, limit), usypy (materiał, tonaż, rozdrobnienie, gęstość), koparki, kruszarka. Przycisk **„Zbuduj trasy”** wyznacza najkrótszą drogę każda koparka → kruszarka i wstawia ją do scenariusza |
| 🎬 Animacja | Cała zmiana odtwarzana przyspieszona na mapie (×60…×1200): wozidła jadą po narysowanych drogach, kolor = stan (jedzie, ładowany, czeka…). Na żywo: koszt narastająco i zł/t, paliwo, tony, kosz kruszarki, kolejki – z porównaniem do przydziału naturalnego **tego samego dnia** |
| ⚙️ Scenariusz | Edycja wszystkich parametrów w tabelach + szybki kalkulator match factor; zapis do YAML |
| 📊 Wyniki zmiany | KPI, koparki, oś czasu wozideł (gdzie czekają), kosz kruszarki i hałda, struktura kosztu, eksport CSV |
| 📍 Podgląd na żywo | Suwak czasu: wozidła na mapie i na profilu trasy, **ekran operatora koparki** (kto pod łyżką, kto czeka, kto nadjeżdża i za ile), **ekrany kierowców** (polecenia), animacja zmiany |
| 🧠 Uczenie RL | Uczenie ze wzmocnieniem polityki przydziału wozideł do koparek na tysiącach symulowanych zmian; wykresy: koszt zł/t, oszczędność vs naturalny, wydajność, paliwo, kolejki, wykorzystanie koparek w kolejnych iteracjach; test końcowy; „Zastosuj politykę RL” |
| ⚖️ Porównanie wariantów | Liczba wozideł × warianty (przydział stały/automatyczny, hałda, doradztwo prędkości) z rekomendacją; porównanie zapisanych scenariuszy |
| 💥 Odstrzał ↔ kruszarka | Mine-to-mill: koszt całkowity na tonę w funkcji rozdrobnienia (droższy odstrzał vs szybszy załadunek i kruszenie) |
| 🤖 Model ETA | Trening modelu czasów przejazdu na danych z wielu zmian, kalkulator ETA |

### Sieć dróg, poziomy, przypisania

- Drogi tworzą **sieć z węzłami**: koniec drogi postawiony ≤ 30 m od innej drogi – także w jej środku – tworzy
  skrzyżowanie. Wspólny odcinek (np. kruszarka → rampa) rysuje się raz, a do usypów tylko odgałęzienia.
- **Rzędne** wszystkich węzłów liczone są od kruszarki (jej rzędna do ustawienia) wzdłuż nachyleń dróg. Odgałęzienie
  w połowie rampy startuje na właściwej wysokości, każda koparka ma swój poziom, a nachylenie każdego odcinka trasy
  wynika z różnicy wysokości w kierunku jazdy (zjazd = minus, wyjazd = plus).
- **Przypisania wozideł** (zakładka Scenariusz → Wozidła): `Koparki` = do których koparek grupa może jeździć;
  `Dwa usypy w kursie` = np. `K-1:0.5, K-2:0.5` – pół ładunku z K-1, przejazd po sieci, pół z K-2, potem do kruszarki.
- **Przydział wozideł**: *naturalny* (obecna praktyka – „swoja” koparka, czasem gdzie indziej, bez informacji),
  *stały*, *heurystyka systemu* (tam, gdzie najszybciej załadują), *polityka RL*.

Scenariusz **„mapa demo”** (`scenarios/kruszyk_mapa_demo.yaml`, generowany przez `tools/make_demo_map.py`) pokazuje
to wszystko na przykładowej, syntetycznej geometrii: droga główna, rampa 9 %, zjazd z połowy rampy na poziom pośredni
(−37,6 m), poziom dolny (−66,2 m) z dwoma usypami, 3 koparki na wapieniu i dolomicie, grupa wozideł jeżdżąca na dwa usypy.

Podkład mapy: zdjęcia satelitarne Esri World Imagery + OpenStreetMap (bez klucza API). Leaflet do animacji jest
dołączony w `kruszyk/vendor` (licencja BSD-2), więc animacja nie wymaga dostępu do CDN.
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
| `kruszyk/rl.py` | Uczenie ze wzmocnieniem polityki przydziału (CEM, wspólne liczby losowe), porównanie polityk, wyjaśnienie wag |
| `kruszyk/anim.py` | Trajektorie wozideł po geometrii dróg + przebieg kosztów minuta po minucie → odtwarzacz HTML/Leaflet |
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
