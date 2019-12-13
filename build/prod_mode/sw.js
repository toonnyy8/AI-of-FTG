console.log('Hello from sw.js')

let CACHE_NAME = "cacheV1"
var urlsToCache = [
    '/',
    './main.fb6bbcaf.js',
    './agent.347b5559.js'
];


self.addEventListener("install", (e) => {
    console.log(e)
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
})

self.addEventListener('fetch', (e) => {
    console.log(e)
    e.respondWith(
        caches.match(e.request)
            .then((response) => {
                // Cache hit - return response
                if (response) {
                    return response;
                }
                return fetch(e.request);
            }
            )
    );
});