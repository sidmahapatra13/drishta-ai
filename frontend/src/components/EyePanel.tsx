import { assetUrl, type EyeResult } from "../api";
import { LesionEvidence } from "./LesionEvidence";
import { QualityGauges } from "./QualityGauges";
import "./EyePanel.css";

/**
 * What the camera captured, and what the system made of it.
 *
 * The photograph is shown even when the image was refused - the health worker
 * needs to see the frame they took in order to take a better one, and the
 * ophthalmologist reviewing the case needs to see what was rejected on their
 * behalf.
 */
export function EyePanel({ eye }: { eye: EyeResult }) {
  const enhanced = eye.quality.enhancement_applied;

  return (
    <section className="eye">
      <header className="eye-head">
        <h3 className="eye-name">{eye.eye} eye</h3>
        {enhanced && <p className="eye-enhanced readout">enhanced · CLAHE</p>}
      </header>

      {eye.image_url && (
        <div className="eye-figures" data-single={!eye.gradcam_url}>
          <figure>
            <img src={assetUrl(eye.image_url)!} alt={`${eye.eye} fundus photograph`} />
            {/* The pipeline stores the enhanced frame when enhancement ran, so
                captioning this "as captured" would be false on exactly the
                cases where the difference matters most. */}
            <figcaption>
              {enhanced ? "After enhancement" : "As captured"}
            </figcaption>
          </figure>
          {eye.gradcam_url && (
            <figure>
              <img src={assetUrl(eye.gradcam_url)!} alt={`${eye.eye} attention map`} />
              <figcaption>
                Regions that influenced the grade. Not proof that a lesion is there.
              </figcaption>
            </figure>
          )}
        </div>
      )}

      {enhanced && (
        <p className="eye-note">
          Enhancement is shown to the person taking the photograph. The grade and
          the attention map are read from the original capture, which is what the
          network was trained on.
        </p>
      )}

      <QualityGauges quality={eye.quality} />
      <LesionEvidence lesions={eye.lesions} />
    </section>
  );
}
