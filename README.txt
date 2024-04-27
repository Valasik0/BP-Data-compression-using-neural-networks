První věc po spuštění aplikace je potřeba vybrat soubor, se kterým chcete pracovat - aplikace je ošetřena a bez načtení souboru nepůjde nic udělat, akorát poskládat model, ale nepůjde spustit učení.

Po načtení souboru se zobrazí jeho základní informace a je možné spočítat jeho entropii pomocí tlačítka Calculate Entropy v levém horním modulu (Text Information). 

Dále je možno poskládat si architekturu modelu s vlastním nastavením a pomocí tlačítka Run spustíte učení modelu na vstupním textu. Při učení se otevře nové okno, kde se zobrazuje průběh učení. Na konci učení se zobrazí graf průběhu loss funce.

Po naučení modelu je možné spočítat komprimovanou velikost načteného souboru pomocí tlačítka Compute v levém modulu (Compressed size) nebo model uložit pro další použití pomocí tlačítka Save. Po stisknutí tlačíta Compute se otevře nové okno s průběhem výpočtu.

Je také možnost načíst model pomocí tlačítka Load model a pomocí něj spočítat komprimovanou velikost souboru. Tento soubor ale musí odpovídat načtenému modelu - musí mít stejný vstupní rozměr, tedy velikost abecedy a délku kontextu.

Při nastavování modelu jsou ošetřeny situace, které nemůžou nastat při sestavování modelu. Pro vyčištění tabulky s architekturou modelu slouží tlačítko koše vlevo dole vedle tabulky. Pro odebrání konkrétní vrstvy klikněte pravým na danou vrstvu a dejte Delete.

Aplikace funguje pro různé typy souborů (obrázky, dokumenty), ale je dělaná především na textové soubory, na kterých dosahuje nejlepších výsledků.

Aplikace je ošetřena na situace, které mouhou nastat a podle hlášek (message boxů) navádí, co je špatně.

Bakalářská práce
Václav Vyrobík
VYR0020

