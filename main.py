import cv2
from imutils import resize
#OpenCV ve imutils kütüphaneleri içe aktarılır.
yuzCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
tambedenCascade = cv2.CascadeClassifier('Cascades/haarcascade_fullbody.xml')
altbedenCascade = cv2.CascadeClassifier('Cascades/haarcascade_lowerbody.xml')
ustbedenCascade = cv2.CascadeClassifier('Cascades/haarcascade_upperbody.xml')
#Yüz, tam beden, alt beden ve üst beden sınıflandırıcıları oluşturulur. Bu sınıflandırıcılar,
# yüz, tam beden, alt beden ve üst beden tespiti yapmak için kullanılacak önceden eğitilmiş Haar
# özellik tabanlı sınıflandırıcı modellerini yükler.
imaj = cv2.imread('resimler/111DSC_1250.jpg', cv2.IMREAD_COLOR)
#imaj değişkenine 'resimler/111DSC_1250.jpg' dosyasından renkli olarak bir görüntü okunur.
if imaj is None:
    print('Resim dosyası bulunamadı!')
else:
    imaj = resize(imaj, width=1000, height=1000)
    gri = cv2.cvtColor(imaj, cv2.COLOR_BGR2GRAY)
#Eğer görüntü okunamazsa hata mesajı yazdırılır. Okunan görüntü varsa, resize fonksiyonuyla görüntü
# boyutu 1000x1000 piksele yeniden boyutlandırılır ve renkli görüntü gri adında bir gri tonlamalı
# görüntüye dönüştürülür.
    yuzler = yuzCascade.detectMultiScale(gri, 1.2, 5)
    tambedenler = tambedenCascade.detectMultiScale(gri, 1.02, 2)
    altbedenler = altbedenCascade.detectMultiScale(gri, 1.02, 3)
    ustbedenler = ustbedenCascade.detectMultiScale(gri, 1.05, 3)
#detectMultiScale yöntemi kullanılarak yüz, tam beden, alt beden ve üst beden bölgeleri tespit edilir.
# Bu yöntem, gri tonlamalı görüntü üzerinde belirli sınıflandırıcıyı kullanarak nesne tespiti yapar.
    for (x,y,w,h) in tambedenler:
        cv2.rectangle(imaj,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(imaj, 'beden', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#Tam beden bölgelerinin bulunduğu dikdörtgenlerin ve etiketlerin çizildiği döngü.
    for (x,y,w,h) in altbedenler:
        cv2.rectangle(imaj,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(imaj, 'alt beden', (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#Alt beden bölgelerinin bulunduğu dikdörtgenlerin ve etiketlerin çizildiği döngü.
    for (x,y,w,h) in ustbedenler:
        cv2.rectangle(imaj,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.putText(imaj, 'üst beden', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#Üst beden bölgelerinin bulunduğu dikdörtgenlerin ve etiketlerin çizildiği döngü.
    cv2.imshow('imaj', imaj)
#Sonuç görüntüsünün 'imaj' başlığıyla ekranda gösterildiği kısım.
    while True:
        k=cv2.waitKey(10) & 0xFF
        if k == 27 or k == ord('q'):
            break
#Kullanıcının programı sonlandırmak için ESC tuşuna veya q tuşuna basmasını bekleyen bir
# döngü. ESC veya q tuşuna basıldığında döngüden çıkılır.
cv2.destroyAllWindows()
#Açık olan tüm pencerelerin kapatıldığı kısım.
