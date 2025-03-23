import React, { useEffect, useRef } from 'react';
import './Map.css';

const Map = () => {
  const mapRef = useRef(null);

  useEffect(() => {
    const loadMap = () => {
      const google = window.google;

      if (google && mapRef.current) {
        const map = new google.maps.Map(mapRef.current, {
          center: { lat: 40.7128, lng: -74.0060 },
          zoom: 12,
        });

        // Simulated route markers
        const routeCoordinates = [
          { lat: 40.7128, lng: -74.0060 },
          { lat: 40.7308, lng: -73.9973 },
          { lat: 40.7580, lng: -73.9855 }
        ];

        routeCoordinates.forEach((coord, index) => {
          new google.maps.Marker({
            position: coord,
            map,
            title: `Point ${index + 1}`
          });
        });

        const routePath = new google.maps.Polyline({
          path: routeCoordinates,
          geodesic: true,
          strokeColor: '#FF5733',
          strokeOpacity: 0.8,
          strokeWeight: 3
        });

        routePath.setMap(map);
      }
    };

    if (!window.google) {
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=YOUR_GOOGLE_MAPS_API_KEY`;
      script.async = true;
      script.onload = loadMap;
      document.body.appendChild(script);
    } else {
      loadMap();
    }
  }, []);

  return (
    <div className="map-container" ref={mapRef} />
  );
};

export default Map;
