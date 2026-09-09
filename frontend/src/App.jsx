import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import NewPatient from "@/pages/NewPatient";

// ModelExplorer/DatasetInspector/Training pages still exist under src/pages/
// but are intentionally unrouted — they predate this project's real rebuild
// and show a superseded architecture + fabricated demo numbers. Re-add routes
// here if/when those pages get rewritten to match the real project.

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/"             element={<Dashboard />} />
        <Route path="/new-patient"  element={<NewPatient />} />
      </Route>
    </Routes>
  );
}