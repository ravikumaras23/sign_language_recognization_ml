import React, {
  useState,
} from "react";

import SignLanguageCamera from "../components/SignLanguageCamera";


function ConvertPage() {

  const [
    currentSign,
    setCurrentSign,
  ] = useState("");


  const handlePrediction = (
    result
  ) => {

    console.log(
      "ML prediction:",
      result
    );

    if (
      result &&
      result.prediction !== null
    ) {

      setCurrentSign(
        result.prediction
      );
    }
  };


  return (
    <div
      style={{
        padding: "30px",
      }}
    >

      <h1>
        Sign Language Detection
      </h1>


      <SignLanguageCamera
        language="ISL"
        onPrediction={
          handlePrediction
        }
      />


      <div
        style={{
          marginTop: "20px",
          fontSize: "28px",
        }}
      >
        Current Sign:{" "}

        <strong>
          {currentSign || "..."}
        </strong>

      </div>

    </div>
  );
}


export default ConvertPage;