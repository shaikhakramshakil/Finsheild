import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Architecture, Dashboard, Investigation, Performance, Privacy } from "./pages";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/investigate/:id" element={<Investigation />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/privacy/:uid" element={<Privacy />} />
      </Routes>
    </BrowserRouter>
  );
}
