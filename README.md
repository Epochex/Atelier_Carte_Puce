Strong Authentication System – Verification & Attack-Resistance README

This repository implements and validates a multi-factor strong authentication system based on:
- Smart card (GemClub-Memo1, storage-only)
- User PIN
- Biometric template binding
- Database-side integrity checks
- Audit logging

Because GemClub-Memo1 is a pure storage card with no cryptographic capability, the system does NOT rely on the secrecy of card data. Security is enforced through strict cross-layer consistency checks.

1. Environment Initialization
```shell
rm -f data/app.db
python3 -m scripts.init_db
```
2. Smart Card Read/Write Sanity Check
```shell
python3 -m scripts.test_card
```
Purpose:
- Ensure the card is detected
- Verify basic read/write capability

3. User Enrollment (Card + PIN + Biometric)
```shell
python3 -m scripts.enroll_user --user-id lin
```
Enrollment actions:
- Enforce PIN policy
- Capture biometric image
- Compute template SHA-256
- Store SHA-256 in DB
- Write tpl_hash8 (first 8 bytes of SHA-256) to card

4. Card ↔ Database Consistency Verification

4.1 Inspect card app_record
```shell
python3 -m scripts.test_card
```
Record:
- tpl_hash8

4.2 Inspect database record
```shell
sqlite3 data/app.db \
"SELECT user_id, template_path, template_sha256 FROM biometrics WHERE user_id='lin';"
```
4.3 Verify biometric file integrity

sha256sum data/templates/lin.png

Requirement:
- First 8 bytes of template SHA-256 MUST match tpl_hash8 on card
- Otherwise authentication is denied

5. Two-Factor Authentication (2FA) Validation

Edit config.yaml:

auth:
  required_factors: 2
  max_pin_attempts: 3
  lockout_seconds: 60

Run authentication:
```shell
python3 -m scripts.demo_run
```
Expected behavior:
- Card + PIN required
- Biometric factor disabled
- Repeated wrong PIN triggers temporary lockout
- Access restored after timeout + correct PIN

6. Audit Log Inspection
```shell
sqlite3 data/app.db \
"SELECT id, ts, card_id, user_id, pwd_ok, bio_score, decision, reason \
 FROM auth_logs ORDER BY id DESC LIMIT 15;"
```
Used to verify:
- Decision logic
- Lockout behavior
- Tamper detection

7. Biometric Template Tamper Detection

Tamper with template file:
```shell
printf 'X' >> data/templates/lin.png
python3 -m scripts.demo_run
```
Expected result:
- DENIED
- reason = template_tampered

Recovery:
```shell
python3 -m scripts.enroll_user --user-id lin
python3 -m scripts.demo_run
```
8. Card Binding Enforcement (Core Security Property)

Authentication strictly enforces the following chain:

[ Physical Card UID ] -> [ Card app_record.card_uid ] -> [ Database user.card_uid ] -> [ Biometric template ownership (user_id) ]

Any mismatch at ANY level results in immediate DENY.

9. Card Binding Attack Scenarios

Attack scenario                          Result without binding     Result with binding
-------------------------------------
Swap card, same user                     Possible bypass            DENIED
Clone app_record to another card         Possible bypass            DENIED
DB user mapped to wrong card             Possible bypass            DENIED
Copy biometric template to another user  Possible bypass            DENIED

10. Threat Model Justification

GemClub-Memo1 characteristics:
- Storage-only smart card
- No cryptographic processor
- No protection against data extraction after PIN unlock

Therefore:
- Card data confidentiality is assumed compromised
- Security is NOT based on secrecy

Security is enforced by:
- Card-side: user_hash8, tpl_hash8
- Database-side: template_sha256
- Mandatory card binding verification at authentication time

Any of the following attacks are detected and rejected:
- Card cloning
- Template swapping
- Database rollback
- Card/user mismatch

11. Card Cloning Attack Demonstration

11.1 Baseline (Legitimate Card A)
```shell
python3 -m scripts.enroll_user --user-id lin
python3 -m scripts.test_card
python3 -m scripts.demo_run
```
Result:
- Authentication succeeds

11.2 Attack Preparation (Clone to Card B)

Remove Card A, insert Card B.

Force enrollment using same user-id and same template:
```shell
python3 -m scripts.enroll_user --user-id lin
python3 -m scripts.test_card
```
Observation:
- Card B contains identical app_record data

11.3 Attack Attempt
```shell
python3 -m scripts.demo_run
```
Expected result:
- DENIED
- Reason: card binding mismatch

