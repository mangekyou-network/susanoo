use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    msg,
    pubkey::Pubkey,
};
use groth16_solana::groth16::Groth16Verifier;
use groth16_solana::errors::Groth16Error;

mod verifying_key;
use verifying_key::VERIFYINGKEY;

// Create our own error wrapper
#[derive(Debug)]
pub enum VerifierError {
    Groth16Error(Groth16Error),
}

// Implement conversion from our error to ProgramError
impl From<VerifierError> for ProgramError {
    fn from(_e: VerifierError) -> Self {
        ProgramError::Custom(1)
    }
}

// Implement conversion from Groth16Error to our error
impl From<Groth16Error> for VerifierError {
    fn from(e: Groth16Error) -> Self {
        VerifierError::Groth16Error(e)
    }
}

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    // Extract proof and public inputs from instruction data
    let (proof_a, rest) = instruction_data.split_at(64);
    let (proof_b, rest) = rest.split_at(128);
    let (proof_c, public_inputs) = rest.split_at(64);

    // Convert proof components
    let proof_a: [u8; 64] = proof_a.try_into().map_err(|_| ProgramError::InvalidInstructionData)?;
    let proof_b: [u8; 128] = proof_b.try_into().map_err(|_| ProgramError::InvalidInstructionData)?;
    let proof_c: [u8; 64] = proof_c.try_into().map_err(|_| ProgramError::InvalidInstructionData)?;

    // Convert public inputs to required format [[u8; 32]; 1]
    // Assuming we expect exactly one public input. Adjust the array size as needed.
    let mut public_input_array: [[u8; 32]; 1] = [[0u8; 32]; 1];
    
    // Check if we have the expected number of inputs
    if public_inputs.len() != 32 {
        return Err(ProgramError::InvalidInstructionData);
    }
    
    public_input_array[0].copy_from_slice(&public_inputs[..32]);

    // Create verifier
    let mut verifier = Groth16Verifier::new(
        &proof_a,
        &proof_b,
        &proof_c,
        &public_input_array,
        &VERIFYINGKEY,
    ).map_err(VerifierError::from)?;

    // Verify the proof
    verifier.verify().map_err(VerifierError::from)?;
    msg!("✅ Proof verified successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification() {
        // Add test with sample proof and public inputs
    }
} 