# 프로젝트 저장·불러오기 MVP

Google 로그인 브랜치의 `firebaseApp`과 `useAuth`를 재사용합니다. 인증 코드는 변경하지 않습니다.

## 사용 방법

1. Google 로그인 후 첫 화면인 블록 캔버스 상단의 **저장·불러오기**를 누릅니다.
2. 이름을 입력하고 **새 프로젝트로 저장**합니다.
3. 이후 **변경 사항 저장**은 같은 문서를 갱신합니다. 새 프로젝트로 저장은 별도 복사본을 만듭니다.
4. 목록의 **불러오기**를 누릅니다. 현재 변경 사항이나 사진·학습 결과가 있으면 화면 안에서 확인을 받습니다.

모델 구성과 설정을 저장합니다. 사진과 학습 결과는 포함되지 않습니다.
블록 ID·위치·설정, 연결 ID·출발·도착, 클래스 ID·이름·순서, 학습 설정, 템플릿 ID를 보존합니다.
v2에서는 VLM 데이터셋의 repo·config·split과 사진/질문/답 열 이름도 보존합니다. 실제 행·미리보기·사진·토큰은 저장하지 않습니다.
불러온 클래스의 사진은 빈 목록이며 학습 결과도 초기화합니다. 학습 중에는 불러올 수 없습니다.

로그아웃하거나 다른 계정으로 바꾸면 프로젝트 목록·이름·현재 문서 ID뿐 아니라 기존 계정의 작업 내용과 사진도 초기화합니다.
필요한 구성은 로그아웃 전에 저장하세요. 로그인 전 작성한 익명 작업은 처음 로그인할 때 유지합니다.
이전 계정의 늦은 요청·학습 결과는 새 계정 화면에 적용하지 않습니다.

## Firebase 콘솔에서 확인할 일

프로젝트·웹 앱·Google 로그인·Firestore 생성과 팀원 초대는 사용자 확인으로 완료됐습니다.
로컬 설정값 네 개와 Git 제외, Firebase 인증 설정 조회 성공, `localhost`와 운영 도메인 승인도 확인했습니다.
실제 규칙은 아직 게시하지 않았다고 사용자에게 확인받았습니다. 실제 데이터 쓰기는 보류합니다.

1. 팀의 기존 Firebase 프로젝트를 확인하고, **프로젝트 설정 → 일반 → 내 앱**에서 웹 앱을 확인하거나 등록합니다.
2. 웹 앱 설정에서 `apiKey`, `authDomain`, `projectId`, `appId`를 확인합니다.
3. 로그인 브랜치의 `.env.example`을 참고하여 로컬 `.env.local`에 실제 값만 넣습니다. `.env.local`은 커밋하지 않습니다.
4. **Authentication → 로그인 방법 → Google**을 활성화하고 지원 이메일을 지정합니다.
5. **Authentication → 설정 → 승인된 도메인**에서 `tensorblocks.netlify.app`, 로컬 검증용 `localhost`를 확인합니다.
6. **Firestore Database → 데이터베이스 만들기**에서 기본 `(default)` 데이터베이스를 만듭니다. Standard edition을 사용하고 위치는 팀과 먼저 결정합니다.
7. **Firestore Database → 규칙**에서 저장소의 `firestore.rules`를 검토합니다. 기존 규칙이 있으면 먼저 비교하고 필요한 경로만 통합합니다. 기존의 공개 허용 규칙이 남아 있으면 새 제한이 무효가 될 수 있습니다.
8. 별도 배포 승인 후 검증한 규칙을 게시합니다. 이번 구현에서는 규칙을 게시하지 않았습니다.
9. 팀원은 **프로젝트 설정 → 사용자 및 권한**에서 필요한 역할로 추가합니다. 앱 사용자의 Google 로그인과 콘솔 관리 권한은 별개입니다.

저장 경로는 `users/{uid}/projects/{projectId}`입니다. 규칙은 본인만 읽기·생성·수정하도록 제한하며 삭제·공유는 허용하지 않습니다.
생성 시각은 생성 때만 기록하고 수정 시각은 서버 시각으로 갱신합니다. 규칙은 최상위 구조와 설정 범위를 검사하고, 클라이언트가 블록별 필드와 연결 대상을 추가 검증합니다.
학습할 수 없는 교육용 오류 모델도 저장할 수 있으므로 그래프의 학습 가능 여부는 저장 조건이 아닙니다.

## 로컬 검증

브라우저 앱을 실행하는 데 Node 24가 필요한 것은 아닙니다. Firebase SDK 개발 저장소뿐 아니라 배포된 `@firebase/firestore`와 규칙 테스트 도구에도 Node 24.12 이상이라는 engines 선언이 있습니다.
브라우저용 Firebase 진입점은 Node용 진입점과 별개입니다. Node 22.23.4에서 별도 폴더의 깨끗한 설치(엔진 경고 있음), 빌드와 기본 자동 테스트 18개가 통과했습니다.
따라서 기존 Netlify Node 22에서 이 정적 앱을 빌드할 수 있음은 확인했습니다. 다만 엔진 엄격 검사가 켜져 있으면 설치가 거부될 수 있으며, Node에서 SDK를 실행하는 테스트·서버 작업에는 공식 선언에 맞는 Node 24.12 이상을 권장합니다.
Node 24.18과 JDK 21.0.13으로 Emulator 통합 테스트까지 23개가 통과했습니다. 전역 Node·PATH·시스템 설정과 기존 Netlify 설정은 변경하지 않았습니다.
테스트의 모듈 대체 기능은 Node의 experimental 기능입니다.

```bash
npm install
npm test
npm run build
```

기본 테스트는 데이터 왕복 복원·VLM 연결·오류·계정 전환·학습 결과 격리와 네 화면의 서버 렌더링을 확인합니다.
서버 렌더링은 실제 브라우저의 WebGL·웹캠·클릭 테스트를 대신하지 않습니다.

실제 보안 규칙과 Firestore SDK의 통합 검증에는 Java와 Firebase CLI가 필요합니다.
이번 검증은 작업 폴더 `.local-tools/java21`에 받은 공식 Temurin JDK 21 ZIP으로 수행했습니다.
Java 실행 경로는 테스트 명령의 프로세스에만 추가했습니다. 재실행 시 JDK 경로를 지정하거나 별도 JDK 21 환경을 준비하세요.

다음 명령은 실제 Firebase 프로젝트 대신 `demo-tensorblocks`라는 로컬 전용 프로젝트와 127.0.0.1:8080을 사용합니다.
Firebase 로그인이나 운영 설정값이 필요하지 않습니다.

```bash
npx firebase-tools emulators:exec --config firebase.emulator.json --only firestore --project demo-tensorblocks "npm test"
```

Emulator가 없으면 보안 규칙 테스트는 이유를 표시하고 건너뜁니다. 건너뜀은 통과가 아닙니다.
이 테스트는 본인 접근 허용, 비로그인·타인 접근 거부, 잘못된 메타데이터 거부, 실제 저장·수정·목록·VLM 조회를 확인합니다.

브라우저에서 가짜 사용자 두 명으로 실제 저장 UI를 확인하려면 아래 두 명령을 각각 터미널에서 실행합니다.

```bash
npx firebase-tools emulators:start --config firebase.emulator.json --only auth,firestore --project demo-tensorblocks
node scripts/emulator-browser.mjs
```

`http://127.0.0.1:5183/tests/browser-emulator.html`을 엽니다.
전용 서버는 `.env.local`을 읽지 않고 demo 설정만 사용합니다. 테스트 Alice/Bob 버튼은 로컬 Auth Emulator에만 사용자를 만듭니다.
실제 Google 로그인 테스트는 별도로 `http://localhost:5182` 같은 승인된 localhost 주소에서 진행합니다. 현재 `127.0.0.1`은 실제 Firebase 승인 도메인에 없습니다.
이 테스트 페이지와 도구는 운영 빌드 진입점에 포함되지 않습니다.

실제 설정의 읽기 전용 점검은 `node scripts/check-firebase-config.mjs`로 재실행할 수 있습니다.
이 도구는 값을 출력하지 않고 인증 설정 조회와 무작위 신규 문서 경로의 비로그인 GET 요청만 보냅니다. 문서를 만들거나 기존 문서를 읽지 않습니다.
HTTP 403은 해당 요청이 거부됐다는 뜻이며, 운영 규칙 전체가 이 저장소 규칙과 일치한다는 증거는 아닙니다.

## Netlify 적용 전

`VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, `VITE_FIREBASE_PROJECT_ID`, `VITE_FIREBASE_APP_ID`를 실제 웹 앱 값으로 등록해야 합니다.
Netlify의 Site/Project configuration → Environment variables에서 빌드 범위와 운영 환경을 확인합니다.
Vite는 빌드 시 값을 넣으므로 설정 후 새 빌드가 필요합니다. 서비스 계정 비공개 키는 절대 넣지 않습니다.
Firebase 웹 설정은 브라우저에 전달되며 실제 접근 제한은 인증과 Firestore 규칙이 담당합니다.

배포 전에는 두 Google 계정으로 저장·수정·새로고침·불러오기·로그아웃을 검증하고,
학습·블록 편집·3D·Colab 내보내기·웹캠도 실제 브라우저에서 확인하세요.
Google 로그인은 `drive.file` 범위도 요청하므로 관련 OAuth 문제는 로그인 담당자와 확인합니다.
협업 규칙은 팀원 1명 이상 Approve 후 PR 작성자가 merge하는 방식입니다. 사용자의 명시적인 승인 없이 merge하거나 main에 직접 push하지 않습니다.
이번 단계에서는 commit·push·PR 생성·규칙 게시·Netlify 배포를 하지 않습니다. Netlify 작업은 배포 담당자에게 전달합니다.

## 범위와 한계

- 새 문서는 `schemaVersion: 2`입니다. v1 문서는 기본 데이터셋 메타데이터를 채워 읽고, 다음 저장 시 v2로 변환합니다. 생성 시각은 유지합니다. 알 수 없는 버전은 거부합니다.
- 운영 규칙도 저장소의 v1/v2 호환 규칙을 검토·적용해야 합니다. 예전 v1 전용 규칙에서는 v2 저장이 거부됩니다.
- 문서 제한에 여유를 두고 직렬화 데이터는 700,000바이트, 블록 500개·연결 1,000개·클래스 100개 이내로 제한합니다.
- 이미지, 썸네일, 브라우저 객체, 실행 함수, 학습 가중치·기록은 업로드하지 않습니다.
- 자동 저장·삭제·공유·사진 저장·다중 탭 동시 편집 충돌 해결은 포함하지 않습니다. 같은 프로젝트를 동시에 저장하면 마지막 저장이 반영됩니다.
- 목록은 해당 사용자의 전체 프로젝트를 서버에서 읽고 수정 시각으로 정렬합니다. 프로젝트가 많아지면 페이지 나누기와 목록 전용 메타데이터 분리가 필요합니다.
- 오프라인에서는 작업을 시작하지 않습니다. 저장 도중 연결이 끊기면 SDK가 재접속을 기다릴 수 있으며 성공 안내는 서버 확인 이후에만 표시합니다.
- `npm audit`의 high 5개는 `@grpc/grpc-js`의 동일 취약점이 Firebase 의존성 경로로 전파된 결과입니다. 브라우저 빌드에 gRPC나 Firestore Node 진입점이 포함되지 않는 것을 `node scripts/check-browser-bundle.mjs`로 확인했습니다. 자세한 영향과 수정 선택지는 `docs/VALIDATION.md`를 참고하세요.
