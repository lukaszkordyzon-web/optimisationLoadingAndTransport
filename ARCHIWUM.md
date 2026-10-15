# Kruszyk – wersja 1 (archiwum)

Znacznik w repozytorium: **`kruszyk-v1-archiwum`**. Wersja 1 to aplikacja `app.py` (Streamlit) z modułami „Analiza zmiany” i
„Planowanie produkcji”. Zostaje nietknięta – można do niej wrócić (`git checkout kruszyk-v1-archiwum`) albo coś z niej
wyjąć. Nowa aplikacja powstaje obok i korzysta z tego samego silnika.

## Co zostaje (silnik – pakiet `kruszyk/`)

| Moduł | Co robi |
|---|---|
| `sim.py` | symulacja zmiany (SimPy): fizyka jazdy, kolejki, jednopasmowe rampy z mijankami i pierwszeństwem, awarie (osobne strumienie losowe na maszynę), przerwy, strzelanie, sygnały jazdy |
| `site.py` | graf dróg z mapy, rzędne (wloty pochylni, rzędne punktów drogi), poziomy, trasy |
| `calibrate.py` | kalibracja z jednej zmiany i z tabeli zmian (średnia + rozrzut) |
| `period.py` | okres zmiana po zmianie: usypy, przejazdy koparek, wiercenie, strzały w obszarach |
| `rl.py` | sygnały jazdy (`train_signals`), dawny pełny przydział RL |
| `compare.py`, `goal.py`, `catalog.py`, `config.py` | porównania na wielu zmianach, cel produkcyjny, katalog maszyn, konfiguracja |

Testy: `pytest tests` (169). Walidacja na kopalni testowej: `tools/walidacja/` (README z wynikami).

## Czego się nauczyliśmy

1. **Symulator ≠ system operacyjny.** Wersja 1 mieszała jedno z drugim. Teraz celem jest symulator (cyfrowy bliźniak),
   który pokazuje **ten sam miesiąc z sygnałami RL i bez**.
2. **Bliźniak jest zmianowy** (kalibracja na raportach zmianowych); miesiąc to kolejne zmiany + plan ludzi (strzały,
   przejazdy koparek) jako dane wejściowe.
3. **RL ma sterować jazdą, nie planem.** Pełny przydział (wybór koparki, odstawianie koparek) uczony na jednej zmianie
   wybierał najbliższy usyp do zera i w miesiącu wychodził drożej. Sygnały jazdy (dokąd – sztygar; jak i kiedy jechać –
   system) na scenariuszu startowym: 6 wozideł −8,8 % zł/t, −30 % paliwa; 4 wozidła ≈ 0.
4. **Porównanie miesięcy musi iść według zamrożonego harmonogramu.** Gdy strzały w obszarach planują się same, inne tempo
   wywozu przesuwa strzały i koparki zostają bez urobku – szum większy niż efekt jazdy. Stąd harmonogram maszyn × dni
   w nowej aplikacji.
5. **Postój wozideł nie może osłabić jednej koparki** (rozjeżdża się skład nadawy).
6. **Pytania tylko wtedy, gdy potrzebne** (np. „dokąd potem” – gdy usyp naprawdę się skończy), bez ścian tekstu.
7. Koszt tony na wielu zmianach = suma kosztów / suma ton (nie średnia z zmian); werdykty testem znaków.
