/**
 * Calculating RA and Dec at zenith (i.e., transiting RA and Dec) for a given location and time.
 */

function julianDate(date) {
  // Convert JS Date (UTC) to Julian Date
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth() + 1; // JS: 0-11 → 1-12
  const day =
    date.getUTCDate() +
    (date.getUTCHours() +
      date.getUTCMinutes() / 60 +
      date.getUTCSeconds() / 3600) /
      24;

  let Y = year;
  let M = month;

  if (M <= 2) {
    Y -= 1;
    M += 12;
  }

  const A = Math.floor(Y / 100);
  const B = 2 - A + Math.floor(A / 4);

  const JD =
    Math.floor(365.25 * (Y + 4716)) +
    Math.floor(30.6001 * (M + 1)) +
    day +
    B -
    1524.5;

  return JD;
}

function greenwichMeanSiderealTimeDegrees(date) {
  const JD = julianDate(date);
  const T = (JD - 2451545.0) / 36525.0; // centuries since J2000.0

  // GMST in seconds (IAU 1982/1994 formula)
  const GMST_sec =
    67310.54841 +
    (876600 * 3600 + 8640184.812866) * T +
    0.093104 * T * T -
    6.2e-6 * T * T * T;

  // Convert seconds to degrees: 360° = 86400s ⇒ 1s = 1/240°
  let GMST_deg = (GMST_sec / 240.0) % 360.0;
  if (GMST_deg < 0) GMST_deg += 360.0;

  return GMST_deg;
}

function localSiderealTimeDegrees(date, longitudeDeg) {
  // longitudeDeg: East positive (e.g., -79.4 for 79.4°W)
  let lst = (greenwichMeanSiderealTimeDegrees(date) + longitudeDeg) % 360.0;
  if (lst < 0) lst += 360.0;
  return lst;
}

function raDecAtZenith(latitudeDeg, longitudeDeg, date = new Date()) {
  const lstDeg = localSiderealTimeDegrees(date, longitudeDeg);

  // RA in hours = LST (deg) / 15
  let raHours = (lstDeg / 15.0) % 24.0;
  if (raHours < 0) raHours += 24.0;

  // Declination at zenith is just your latitude
  const decDeg = latitudeDeg;

  return {
    raHours, // 0–24 hours
    decDeg,  // = latitude
  };
}